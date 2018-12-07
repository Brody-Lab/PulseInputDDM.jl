module gaussian_neural_observation

using global_functions, Optim, LineSearches
using Distributions

const dimz = 5

export do_optim_ΔLR, do_optim

function do_optim(pz,py,pstd,std0,dt,data,f_str,N;n::Int=103,
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    inv_map_pz!(pz,dt)     
    inv_map_py!.(py,f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p_opt = vcat(pz,vcat(py...),pstd)

    ###########################################################################################
    ## Optimize

    ll(x) = ll_wrapper(x, data, f_str, N, std0, n=n, dt=dt);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
                iterations = 1000, store_trace = true, 
                show_trace = true, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    pz,py,pstd = breakup(p_opt,N)
    
    map_pz!(pz,dt)       
    map_py!.(py,f_str)
    
    return pz, py, pstd
    
end

function breakup{TT}(p::Vector{TT},N)
    
    pz = p[1:dimz];
    py = reshape(p[dimz+1:N*3+dimz],3,:)
    py = map(i->py[:,i],1:size(py,2))
    
    pstd = p[N*3+dimz+1:end];
    
    return pz,py,pstd
    
end

function map_pz!(x,dt)
    
    x[[1,4,5]] = exp.(x[[1,4,5]]);    
    x[2] = 2. + exp(x[2]);   
    x[3] = -1./(2*dt) + (1./dt)*(0.5*(1.+tanh(x[3])));
    
    return x
    
end

function inv_map_pz!(x,dt)
    
    x[[1,4,5]] = log.(x[[1,4,5]]);    
    x[2] = log(x[2]-2.);    
    x[3] = atanh((2 .* dt * (x[3] + 1./(2.*dt))) - 1.);
    
    return x
    
end

function ll_wrapper{TT}(p_opt::Vector{TT}, data::Dict, f_str::String,
        N,std0;map_str::String="exp",n::Int=203, dt::Float64=2e-2)
        
    pz,py,pstd = breakup(p_opt,N)
    
    map_pz!(pz,dt)       
    map_py!.(py,f_str)

    -(sum(LL_all_trials(pz, py, pstd, std0, data, f_str, N, n=n, dt=dt)) - LLprior(pz[2]))
                 
end

LLprior(B)=1e0*(B - 10).^2

function LL_all_trials{TT}(pz::Vector{TT},py::Vector{Vector{TT}},  
        pstd::Vector{TT},std0::Vector{Float64},data::Dict, f_str::String, N;  
        dt::Float64=2e-2, comp_posterior::Bool=false, n::Int=203)
        
    #break up latent variables
    vari,B,lambda,vara = pz[1:4]
   
    # spatial bin centers, width and edges
    xc,dx,xe = bins(B,n=n)
    
    # make initial latent distribution
    P = P0(vari,zero(TT),n,dx,xc);
   
    M = zeros(TT,n,n);
    M!(M,vara*dt,lambda,zero(TT),dx,xc,n=n,dt=dt)

    #muy = map(p->fy(xc,p,f_str),py)
    #muy = reshape(vcat(muy...),n,:); 
    muy = fy.(xc,py',f_str)
              
    output = pmap((L,R,T,N,y) -> LL_single_trial(pz, pstd[N], std0[N],
            P, M, dx, xc,
            L, R, T, muy[:,N],y,
            comp_posterior=comp_posterior,n=n,dt=dt),
            data["leftbups"], data["rightbups"], data["nT"],
            data["N"], data["spike_counts"])        

end

function LL_single_trial{TT}(pz::Vector{TT}, pstd::Vector{TT}, std0::Vector{Float64},
        P::Vector{TT}, M::Array{TT,2}, dx::TT, xc::Vector{TT},
        L::Vector{Float64}, R::Vector{Float64}, T::Int,
        muy::Array{TT,2},y::Vector{Vector{Float64}};
        comp_posterior::Bool=false, n::Int=203, dt::Float64=2e-2)
    
    #break up parameters
    lambda,vara,vars = pz[3:5]
    
    L,R = binLR(T,L,R;dt=dt)
    
    y = reshape(vcat(y...),:,length(y));

    F = zeros(M)           
    
    c = Vector{TT}(T)
    comp_posterior ? alpha = Array{Float64,2}(n,T) : nothing

    @inbounds for t = 1:T

        sumLR,mu = L[t]+R[t], -L[t]+R[t]

        sumLR > zero(vars) ? (M!(F,vars*sumLR+vara*dt,lambda,mu/dt,dx,xc,n=n,dt=dt); P  = F * P;) : P = M * P
        
        #this is still wonky...10/23
        #arg = sum(log.(1./sqrt.(2*pi*(std0.^2+pstd.^2)))) - 
        #            sum(broadcast(/,(broadcast(-,y[t,:],muy')).^2,2*(std0.^2+pstd.^2))',2);
        arg = gauss_LL.(y[t,:],muy',sqrt.(pstd.^2+std0.^2))
        #protect against NaNs in gradient
        arg[arg .< log(1e-150)] = log(1e-150);
        #P .*= vec(exp.(arg));
        P .*= vec(exp.(sum(arg,1))); 

        c[t] = sum(P)
        P /= c[t] 

        comp_posterior ? alpha[:,t] = P : nothing

    end

    if comp_posterior

        beta = zeros(Float64,n,T)
        P = ones(Float64,n); #initialze backward pass with all 1's
    
        beta[:,T] = P;

        @inbounds for t = T-1:-1:1
            
            #this is still wonky...10/23
            #arg = sum(log.(1./sqrt.(2*pi*(std0.^2+pstd.^2)))) - 
            #        sum(broadcast(/,(broadcast(-,y[t+1,:],muy')).^2,2*(std0.^2+pstd.^2))',2);
            arg = gauss_LL.(y[t+1,:],muy',sqrt.(pstd.^2+std0.^2))
            #protect against NaNs in gradient
            arg[arg .< log(1e-150)] = log(1e-150);
            #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),1)));
            P .*= vec(exp.(sum(arg,1))); 
            
            sumLR,mu = L[t+1]+R[t+1], -L[t+1]+R[t+1]

            sumLR > zero(vars) ? (M!(F,vars*sumLR+vara*dt,lambda,mu/dt,dx,xc,n=n,dt=dt); P  = F' * P;) : P = M' * P

            P /= c[t+1] 

            beta[:,t] = P

        end

    end

    comp_posterior ? (return alpha .* beta) : (return sum(log.(c)))

end

function do_optim_ΔLR(py::Vector{Vector{Float64}},pstd::Vector{Float64},data,
        f_str::String,std0::Vector{Float64}=Vector{Float64}();dt::Float64=1e-3)
    
    ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=dt),data["nT"],data["leftbups"],data["rightbups"]);
    
    inv_map_py!.(py,f_str)
    p = map((x,y)->cat(1,x,y),py,pstd)
    
    pystar = pmap((p,x,trials,std0)->opt_func(p,x,trials,ΔLR,f_str,std0),
        p,data["spike_counts"],data["trial"],std0);
    
    pstdstar = map(x->x[4],pystar);
    pystar = map(x->x[1:3],pystar)
    
    map_py!.(pystar,f_str)
    
    return pystar, pstdstar
    
end

opt_func(p0,x,trials,ΔLR,f_str::String,std0::Float64) = Optim.minimizer(optimize(p0 -> 
        ll_wrapper_ΔLR(p0,x,ΔLR[trials],f_str,std0), 
        p0, method = Optim.BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), 
        linesearch = BackTracking()), autodiff=:forward, g_tol = 1e-12, x_tol = 1e-16, 
        f_tol = 1e-16, iterations = Int(1e16), show_trace = false, allow_f_increases = true));
    
function ll_wrapper_ΔLR{TT}(p::Vector{TT}, x::Vector{Vector{Float64}}, 
        ΔLR::Vector{Vector{Int}},f_str::String, std0::Float64)
            
        pstd,py = p[4],p[1:3]
    
        map_py!(py,f_str)
        #this is still wonky...10/23
        mu = fy.(vcat(ΔLR...),[py],f_str);
    
        #compute LL for all trials, for this neuron
        -sum(gauss_LL.(vcat(x...),mu,sqrt(pstd^2+std0^2)))
            
end
    
gauss_LL(x,mu,sigma) = log(1/sqrt(2*pi*sigma^2)) - (x-mu)^2/(2*sigma^2);

samples(mu,std) = rand(Normal(mu,std))
        
end
