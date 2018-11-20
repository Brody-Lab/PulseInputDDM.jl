module module_DDM_v4

const dt,dims = 2e-2,Dict("z"=>8,"d"=>1,"y"=>4)

import ForwardDiff, Base.convert
using StatsBase, LsqFit, Distributions
using Optim, LineSearches
using helpers

export LL_all_trials, LL_single_trial, convert_data!, group_params, break_params, bins
export map_func!, inv_map_func!, my_sigmoid, Mprime!, make_adapted_clicks
export ll_wrapper, fit_func, qfind, package_data!, compute_x0, sample_latent, compute_priors
export ll_wrapper_diffLR
export dt,dims, diffLR
export x0_spikes, x0_model, map_func_fr!, inv_map_func_fr!

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
convert(::Type{Int},x::ForwardDiff.Dual) = Int(x.value)

function my_sigmoid(x,p)
    
    #y = broadcast(+,p[:,1]',broadcast(/,p[:,2]',(1. + 
    #            exp.(broadcast(*,p[:,3]',broadcast(+,-x,broadcast(/,p[:,4]',p[:,3]')))))));
    #temp = broadcast(*,p[:,3]',broadcast(+,-x,broadcast(/,p[:,4]',p[:,3]')))
    
    #original way
    y = broadcast(+,p[:,1]',broadcast(/,p[:,2]',(1. + exp.(broadcast(+,broadcast(*,-p[:,3]',x),p[:,4]')))));
    temp = broadcast(+,broadcast(*,-p[:,3]',x),p[:,4]')
    
    y[exp.(temp) .<= 1e-150] = broadcast(+,p[:,1]',broadcast(/,p[:,2]',ones(length(x),)))[exp.(temp) .<= 1e-150]
    y[exp.(temp) .>= 1e150] = broadcast(*,p[:,1]',ones(length(x),))[exp.(temp) .>= 1e150]   
    
    return y
    
end

function compute_priors(model_type,N)

    beta = [0., 0., 0., 0., 0., 0., 0., 0.]
    #beta = [0., 0., 1e-1, 0., 0., 0., 0., 0.]
    mu0 =  [0., 0., 10., 0., 0., 0., 0., 0.]

    if any(model_type .== "choice")
        beta = cat(1,beta,0.)
        mu0 = cat(1,mu0,0.)
    end

    if any(model_type .== "spikes")
       
        beta_y = Array{Float64,2}(N,4)
        mu0_y = Array{Float64,2}(N,4)

        for j = 1:N
            #these are what I started using after meeting with Carlos
            #beta_y[j,:] = [1e2,1e1,1e1,1e1];;
            #mu0_y[j,:] = [1e-2,1e1,0.,0.];
            #beta_y[j,:] = [0.,1e-2,1e-1,1e-1]; #I used these values in push before carlos meeting on 7/10
            #these were the last i used on friday before back to carlos methods
            beta_y[j,:] = [0.,0.,0.,0.];
            #beta_y[j,:] = [0.,1e-1,1e-1,1e-1];
            mu0_y[j,:] = [1e-2,1e-2,0.,0.];
            #mu0_y[j,:] = [0.,0.,0.,0.]
            #needed to do this with new c/d because of divide by 0
            #mu0_y[j,:] = [1e-2,1e-2,1e-2,-1e-2];
        end

        beta = cat(1,beta,vec(beta_y))
        mu0 = cat(1,mu0,vec(mu0_y))

    end

    return beta,mu0

end

function x0_model(x,p,kind)
    
   p = map_func_fr(p,kind)
    
   #y = exp(p[1]) + exp(p[2])./(1. + exp.(-p[3] .* x + p[4]))
    
   #y = p[1] + p[2]./(1. + exp.(p[3]*(-x + (p[4]/p[3]))))
   #temp = p[3]*(-x + (p[4]/p[3]))
    
   y = p[1] + p[2]./(1. + exp.(-p[3] .* x + p[4]))
   temp = -p[3]*x + p[4]
    
   y[exp.(temp) .<= 1e-150] = p[1]+p[2]
   y[exp.(temp) .>= 1e150] = p[1]        

   return y

end

function map_func_fr!{TT}(p::Vector{TT},kind::String)
        
    if kind == "exp"
        p[1:2] = exp.(p[1:2])
        p[3:4] = p[3:4]
    elseif kind == "tanh"
        p[1:2] = 1e-5 + 99.99 * 0.5*(1+tanh.(p[1:2]))
        p[3:4] = -9.99 + 9.99*2 * 0.5*(1+tanh.(p[3:4]))
    end
    
    return p
    
end

function inv_map_func_fr!{TT}(p::Vector{TT},kind::String)
    
    if kind == "exp"
        p[1:2] = log.(p[1:2])
        p[3:4] = p[3:4]
    elseif kind == "tanh"
        p[1:2] = atanh.(((p[1:2] - 1e-5)/(99.99*0.5)) - 1)
        p[3:4] = atanh.(((p[3:4] + 9.99)/(9.99*2*0.5)) - 1)
    end
    
    return p
    
end

function x0_spikes(data,N,betas,mu0,kind;dt::Float64=2e-2)
    
    tri = length(data["T"]);
    temp = Vector{Array{Float64,2}}(N)

    betas_y = reshape(betas[end-N*dims["y"]+1:end],N,dims["y"])
    mu0_y = reshape(mu0[end-N*dims["y"]+1:end],N,dims["y"])

    for i = 1:N
        temp[i] = Array{Float64,2}(0,2)
    end

    #loop over trials
    for i = 1:tri

        #compute the cumulative diff of clicks
        t = 0:dt:data["nT"][i]*dt;
        L = fit(Histogram,data["leftbups"][i],t,closed=:left)
        R = fit(Histogram,data["rightbups"][i],t,closed=:left)
        diffLR = cumsum(-L.weights + R.weights)

        for j = 1:length(data["N"][i])
            temp[data["N"][i][j]] = cat(1,temp[data["N"][i][j]],
                cat(2,diffLR,data["spike_counts"][i][:,j]/dt))
        end

    end

    #model(x,p) = exp(p[1]) + exp(p[2])./(1. + exp.(-p[3] .* x + p[4]))

    x0y = Array{Float64,2}(N,4)

    for j = 1:N

        #fit = curve_fit(model, temp[j][:,1], temp[j][:,2], p0)
        #x0y[j,:] = fit.param
        
        #p0 = [1e-2,10.,0.,0.]
        #p0 = [1e-2,1e-2,0.,0.];
        p0 = mu0_y[j,:];
        p0 = inv_map_func_fr(p0,kind);
        
        #was doing this but seems like an error (although fits in fit_analysis came back OK).
        #is poisson eerror function the same?
        #x0_C(p) = sum((x0_model(temp[j][:,1],p) - temp[j][:,2]).^2) + sum(betas_y[j,:] .* (p-(mu0_y[j,:])).^2)
        x0_C(p) = sum((x0_model(temp[j][:,1],p,kind) - temp[j][:,2]).^2) + sum(betas_y[j,:] .* (p-inv_map_func_fr(mu0_y[j,:],kind)).^2)
        
        od = OnceDifferentiable(x0_C,p0; autodiff=:forward)
            results = optimize(od, p0, BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
            linesearch = BackTracking()), Optim.Options(g_tol = 1e-12, x_tol = 1e-16, 
            f_tol = 1e-16, allow_f_increases = true))   
        
        x0y[j,:] = results.minimizer;          
        x0y[j,:] = map_func_fr(x0y[j,:],kind);

    end
    
    return x0y

    #x0_C(p) = sum((x0_model(temp[j][:,1],p) - temp[j][:,2]).^2) + sum(betas_y[j,:] .* (p-mu0_y[j,:]).^2)

    #betas_y = reshape(betas[end-N*dims["y"]+1:end],N,dims["y"])
    #mu0_y = reshape(mu0[end-N*dims["y"]+1:end],N,dims["y"])

    #ll_diffLR(x) = ll_wrapper_diffLR(x, data, betas_y, mu0_y)

    #py = repmat(p0,N,1);

    #od = OnceDifferentiable(ll_diffLR, py; autodiff=:forward)

    #results = optimize(od, py, 
    #    BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
    #    linesearch = BackTracking()), Optim.Options(g_tol = 1e-12, x_tol = 1e-16, 
    #    f_tol = 1e-16, allow_f_increases = true))   

    #x0y = reshape(results.minimizer,N,dims["y"])
    #x0y[:,1:2] = exp.(x0y[:,1:2]);
    
end

#function compute_x0(data,model_type,N;p0::Vector{Float64}=[log(0.+eps()),log(100.),0.,0.])
function compute_x0(data,model_type,N,betas,mu0,kind)

    x0 = [1e-5, 0.+1e-6, 20., 1e-3, 10., 1., 1.-1e-6, 0.2]
    #x0 = [log(1e-5), 0.+1e-6, 20., 1e-3, log(10.), log(1.), 1.-1e-6, 0.2]
    #x0 = [sqrt(1e-5), 0.+1e-6, 20., 1e-3, sqrt(10.), sqrt(1.), 1.-1e-6, 0.2]

    if any(model_type .== "choice")
        x0 = cat(1,x0,0.+1e-6)
    end

    if any(model_type .== "spikes")

        x0y = x0_spikes(data,N,betas,mu0,kind)
        x0 = cat(1,x0,vec(x0y))

    end

    return x0

end

function fit_func(model_type::Union{Array{String},String},N::Int)

    #          vari       inatt          B    lambda       vara    vars     phi    tau_phi 
    fit_vec = [falses(1);falses(1);    trues(4);                         trues(2)];

    any(model_type .== "choice") ? fit_vec = cat(1,fit_vec,trues(1)) : nothing
    any(model_type .== "spikes") ? fit_vec = cat(1,fit_vec,trues(dims["y"]*N)) : nothing
    
    return fit_vec

end

function inv_map_func!{TT}(x::Vector{TT}, model_type::Union{Array{String},String},kind; N::Int64=0)

    x[[1,5,6]] = log.(x[[1,5,6]]);
    #x[[1,5,6]] = x[[1,5,6]];
    x[2] = atanh(2.*x[2]-1.);
    if kind == "exp"
        x[3] = log(x[3]-2.);
    elseif kind == "tanh"
        x[3] = atanh.(((x[3] - 2.)/(100*0.5))-1)
    end
    x[4] = atanh((2 .* dt * (x[4] + 1./(2.*dt))) - 1.);
    x[7] = log(x[7]);
    x[8] = log(x[8]);

    #any(model_type .== "choice") ? x[9] = atanh((2./(2.*2.) * (x[9] + 2)) - 1.) : nothing
     any(model_type .== "choice") ? x[9] = x[9] : nothing
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
           
        if kind == "exp"
            x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N] = log.(x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N]);
            x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N] = x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N];
        elseif kind == "tanh"
        
        x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N] = atanh.(((x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N]-1e-6)/(100*0.5)) - 1);
        x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N] = atanh.(((x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N] + 10.)/(20*0.5))-1);
            
        end
                    
    elseif any(model_type .== "spikes")
        
        if kind == "exp"
        x[dims["z"]+1:dims["z"]+2*N] = log.(x[dims["z"]+1:dims["z"]+2*N]);
        x[dims["z"]+2*N+1:dims["z"]+4*N] = x[dims["z"]+2*N+1:dims["z"]+4*N];
        elseif kind == "tanh"
        
        x[dims["z"]+1:dims["z"]+2*N] = atanh.(((x[dims["z"]+1:dims["z"]+2*N] - 1e-6)/(100*0.5)) - 1);
        x[dims["z"]+2*N+1:dims["z"]+4*N] = atanh.(((x[dims["z"]+2*N+1:dims["z"]+4*N]  + 10.)/(20*0.5))-1);
        end
                
    end
    
    return x

end

function map_func!{TT}(x::Vector{TT}, model_type::Union{Array{String},String},kind; N::Int64=0)

    x[[1,5,6]] = exp.(x[[1,5,6]]);
    #x[[1,5,6]] = x[[1,5,6]];
    x[2] = 0.5*(1+tanh(x[2]));
    if kind == "exp"
    x[3] = 2. + exp(x[3]);
    elseif kind == "tanh"
    x[3] = 2 + 100 * 0.5*(1+tanh.(x[3]))
    end
    x[4] = -1./(2*dt) + (1./dt)*(0.5*(1.+tanh(x[4])));
    x[7] = exp(x[7]);
    x[8] = exp(x[8]);

    #any(model_type .== "choice") ? x[9] = -2 + 2*2*0.5*(1 + tanh(x[9])) : nothing
    any(model_type .== "choice") ? x[9] = x[9] : nothing
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
         
        if kind == "exp"
        x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N] = exp.(x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N]);
        x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N] = x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N];
        elseif kind == "tanh"
        
        x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N] = 1e-6 + 100 * 0.5*(1+tanh.(x[dims["z"]+dims["d"]+1:dims["z"]+dims["d"]+2*N]));
        x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N] = -10 + 20 * 0.5*(1+tanh.(x[dims["z"]+dims["d"]+2*N+1:dims["z"]+dims["d"]+4*N]));
        end
        
    elseif any(model_type .== "spikes")
        
        if kind == "exp"
        x[dims["z"]+1:dims["z"]+2*N] = exp.(x[dims["z"]+1:dims["z"]+2*N]);
        x[dims["z"]+2*N+1:dims["z"]+4*N] = x[dims["z"]+2*N+1:dims["z"]+4*N];
        elseif kind == "tanh"
        
        x[dims["z"]+1:dims["z"]+2*N] = 1e-6 + 100 * 0.5*(1+tanh.(x[dims["z"]+1:dims["z"]+2*N]));
        x[dims["z"]+2*N+1:dims["z"]+4*N] = -10 + 20 * 0.5*(1+tanh.(x[dims["z"]+2*N+1:dims["z"]+4*N]));
        end
                    
    end
    
    return x

end

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, 
        data::Dict, beta::Vector{Float64}, mu0::Vector{Float64}, model_type::Union{String,Array{String}}, kind; 
        N::Int64=0, n::Int=203)
    
    p = group_params(p_opt, p_const, fit_vec)
    p = map_func!(p,model_type,kind,N=N)
    p_opt,p_const = break_params(p, fit_vec)
    
    LL = LL_all_trials(p_opt, p_const, fit_vec, data, beta, mu0, model_type, N=N, n=n)
    
    return LL

end

function make_adapted_clicks(leftbups, rightbups, phi, tau_phi)

    L = ones(typeof(phi),size(leftbups));
    R = ones(typeof(phi),size(rightbups));

    #if phi !== 1.

    # magnitude of stereo clicks set to zero
    if ~isempty(leftbups) && ~isempty(rightbups) && abs(leftbups[1]-rightbups[1]) < eps()
        L[1] = eps()
        R[1] = eps()
    end

        if length(leftbups) <= 1
            ici_l = [];
        else
            ici_L = (leftbups[2:end]  - leftbups[1:end-1])'
        end

        if length(rightbups) <= 1
            ici_R = []
        else
            ici_R = (rightbups[2:end]  - rightbups[1:end-1])'
        end

        for i = 2:length(leftbups)
            if abs(1. - L[i-1]*phi) <= 1e-150
                L[i] = 1.
            else
                last_L = tau_phi*log(abs(1-L[i-1]*phi))
                L[i] = 1 - exp((-ici_L[i-1] + last_L)/tau_phi)
            end
        end;

        for i = 2:length(rightbups)
            if abs(1. - R[i-1]*phi) <= 1e-150
                R[i] = 1.
            else
                last_R = tau_phi*log(abs(1-R[i-1]*phi))
                R[i] = 1 - exp((-ici_R[i-1] + last_R)/tau_phi)
            end
        end;

    #end

        L = real(L)
        R = real(R)

    return L, R

end

function LL_single_trial{TT}(lambda_drift::TT, vara::TT, vars::TT, phi::TT, tau_phi::TT,
        P::Vector{TT}, M::Array{TT,2}, dx::TT, xc::Vector{TT}, model_type::Union{String,Array{String}},
        L::Union{Array{Float64},Float64}, R::Union{Array{Float64},Float64}, T::Int,
        hereL::Union{Array{Int},Int}, hereR::Union{Array{Int},Int};
        lambda::Union{Array{TT},Array{Float64}}=Array{TT}(0,0),
        spike_counts::Union{Array{TT},Array{Int}}=Array{Int}(0,0),
        nbinsL::Union{TT,Int}=1, Sfrac::Union{Float64,TT}=one(TT), pokedR::Union{Bool,TT}=false, 
        comp_posterior::Bool=false, n::Int=203)

    La, Ra = make_adapted_clicks(L,R,phi,tau_phi)

    F = zeros(M)

    if any(model_type .== "choice")
        notpoked = convert(TT,~pokedR); poked = convert(TT,pokedR)
        Pd = vcat(notpoked * ones(nbinsL), notpoked * Sfrac + poked * (one(Sfrac) - Sfrac), poked * ones(n - (nbinsL + 1)))
    end
    
    if any(model_type .== "spikes") 
        Py = exp.(broadcast(-, broadcast(-, spike_counts *  log.(lambda'*dt), sum(lambda,2)' * dt), 
                sum(lgamma.(spike_counts + 1),2)))' 
    end
    
    c = Vector{TT}(T)
    comp_posterior ? alpha = Array{Float64,2}(n,T) : nothing

    @inbounds for t = 1:T
        
        any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
        any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)

        var = vars * (sL + sR);  mu = -sL + sR

        ((sL + sR) > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_drift,mu/dt,dx,xc,n=n); P  = F * P;) : P = M * P
        
        any(model_type .== "spikes") && (P .*= Py[:,t])
        (any(model_type .== "choice") && t == T) && (P .*=  Pd)

        c[t] = sum(P)
        P /= c[t] 

        comp_posterior ? alpha[:,t] = P : nothing

    end

    if comp_posterior

        beta = zeros(Float64,n,T)
        P = ones(Float64,n); #initialze backward pass with all 1's
    
        beta[:,T] = P;

        @inbounds for t = T-1:-1:1

            any(model_type .== "spikes") && (P .*= Py[:,t+1])
            any(model_type .== "choice") && t + 1 == T && (P .*=  Pd)
        
            any(t+1 .== hereL) ? sL = sum(La[t+1 .== hereL]) : sL = zero(phi)
            any(t+1 .== hereR) ? sR = sum(Ra[t+1 .== hereR]) : sR = zero(phi)

            var = vars * (sL + sR);  mu = -sL + sR

            (var > zero(vars)) ? (Mprime!(F,var+vara*dt,lambda_drift,mu/dt,dx,xc,n=n); P  = F' * P;) : P = M' * P

            P /= c[t+1] 

            beta[:,t] = P

        end

    end

    comp_posterior ? (return alpha .* beta) : (return sum(log.(c)))

end

function group_params{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1})
    
    p = Array{TT}(length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    return p
    
end

function break_params{TT}(p::Vector{TT}, fit_vec::BitArray{1})
    
    p_opt = p[fit_vec];
    p_const = Array{Float64}(p[.!fit_vec]);
    
    return p_opt, p_const
    
end

function bins{TT}(B::TT;n::Int=203)
    
    # binning
    dx = 2.*B/(n-2);  #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2))); #centers
    xe = cat(1,xc[1]-dx/2,xc+dx/2) #edges
    
    return xc, dx, xe
    
end

function LL_all_trials{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::BitArray{1}, 
        data::Dict, beta::Vector{Float64}, mu0::Vector{Float64}, model_type::Union{String,Array{String}}; 
        N::Int64=0, comp_posterior::Bool=false, n::Int=203)
    
    p = group_params(p_opt,p_const,fit_vec)
    
    vari,inatt,B,lambda_drift,vara,vars,phi,tau_phi = p[1:dims["z"]]
    #log_vari,inatt,B,lambda_drift,log_vara,log_vars,phi,tau_phi = p[1:dims["z"]]
    #vari,vara,vars = exp(log_vari),exp(log_vara),exp(log_vars)

    #stdi,inatt,B,lambda_drift,stda,stds,phi,tau_phi = p[1:dims["z"]]
    #vari,vara,vars = stdi^2,stda^2,stds^2
    
    any(model_type .== "choice") ? bias = p[9] : zero(TT)

    if any(model_type .== "spikes") & any(model_type .== "choice")
        py = reshape(p[dims["z"]+dims["d"]+1:end],N,4)
    elseif any(model_type .== "spikes")
        py = reshape(p[dims["z"]+1:end],N,4)
    else      
        py = Array{TT,2}(0,4)        
    end

    ntrials = length(data["T"])
   
    # binning
    xc,dx,xe = bins(B,n=n)
    
    # make initial delta function
    P = zeros(xc); P[[1,n]] = inatt/2.; P[ceil(Int,n/2)] = one(TT) - inatt; 
    # Convolve initial delta with vari
    M = zeros(TT,n,n);  Mprime!(M,vari,zero(TT),zero(TT),dx,xc,n=n); P = M * P
    #make ntrial copys for all the trials
    P = pmap(i->copy(P),1:ntrials); 
   
    # build state transition matrix for no input time bins
    Mprime!(M,vara*dt,lambda_drift,zero(TT),dx,xc,n=n)

    if any(model_type .== "choice")
        #nbinsL = ceil(Int,(B+bias)/dx)
        #Sfrac = one(dx)/dx * (bias - (-(B+dx)+nbinsL*dx))
        nbinsL = sum(bias .> xe[2:n])
        Sfrac = (bias - xe[nbinsL+1])/dx
        Sfrac < zero(Sfrac) ? Sfrac = zero(Sfrac) : nothing
        Sfrac > one(Sfrac) ? Sfrac = one(Sfrac) : nothing
    end

    if any(model_type .== "spikes")              
        lambda = my_sigmoid(xc,py)
        #temp = broadcast(+,broadcast(*,-py[:,3]',xc),py[:,4]')
        #lambda[exp.(temp) .<= 1e-150] = broadcast(+,py[:,1]',broadcast(/,py[:,2]',ones(n,)))[exp.(temp) .<= 1e-150]
        #lambda[exp.(temp) .>= 1e150] = broadcast(*,py[:,1]',ones(n,))[exp.(temp) .>= 1e150]        
    end
    
    if any(model_type .== "spikes") & any(model_type .== "choice")
        
        output = pmap((P,L,R,T,nL,nR,N,SC,pokedR) -> LL_single_trial(lambda_drift, vara, vars, phi, tau_phi,
            P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[:,N],spike_counts=SC,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n),
            P, data["leftbups"], data["rightbups"], data["nT"], data["hereL"], data["hereR"],
            data["N"],data["spike_counts"],
            data["pokedR"])
            
    elseif any(model_type .== "choice") 
        
        output = pmap((P,L,R,T,nL,nR,pokedR) -> LL_single_trial(lambda_drift, vara, vars, phi, tau_phi,
            P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            nbinsL=nbinsL,Sfrac=Sfrac,pokedR=pokedR,
            comp_posterior=comp_posterior,n=n),
            P, data["leftbups"], data["rightbups"], data["nT"], data["hereL"], data["hereR"],
            data["pokedR"])
        
    elseif any(model_type .== "spikes")
        
        output = pmap((P,L,R,T,nL,nR,N,SC) -> LL_single_trial(lambda_drift, vara, vars, phi, tau_phi,
            P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            lambda=lambda[:,N],spike_counts=SC,
            comp_posterior=comp_posterior,n=n),
            P, data["leftbups"], data["rightbups"], data["nT"], data["hereL"], data["hereR"],
            data["N"],data["spike_counts"])        
    else
        
        output = pmap((P,L,R,T,nL,nR) -> LL_single_trial(lambda_drift, vara, vars, phi, tau_phi,
            P, M, dx, xc, model_type,
            L, R, T, nL, nR,
            comp_posterior=comp_posterior,n=n),
            P, data["leftbups"], data["rightbups"], data["nT"], data["hereL"], data["hereR"])

    end

    comp_posterior ? (return output) : (return -(sum(output) - sum(beta .* (p_opt - mu0).^2)))

end

function Mprime!{TT}(F::AbstractArray{TT,2},vara::TT,lambda::TT,h::TT,dx::TT,xc::Vector{TT}; n::Int=203)
    
    F[1,1] = one(TT); F[n,n] = one(TT); F[:,2:n-1] = zero(TT)

    ndeltas = max(70,ceil(Int, 10.*sqrt(vara)/dx));

    (ndeltas > 1e3 && h == zero(TT)) ? (println(vara); println(dx); println(ndeltas)) : nothing

    #deltas = collect(-ndeltas:ndeltas) * (5.*sqrt(vara))/ndeltas;
    #ps = broadcast(exp, broadcast(/, -broadcast(^, deltas,2), 2.*vara)); ps = ps/sum(ps);
    
    deltaidx = collect(-ndeltas:ndeltas);
    deltas = deltaidx * (5.*sqrt(vara))/ndeltas;
    ps = exp.(-0.5 * (5*deltaidx./ndeltas).^2); ps = ps/sum(ps);
    
    @inbounds for j = 2:n-1

        abs(lambda) < 1e-150 ? mu = xc[j] + h * dt : mu = exp(lambda*dt)*(xc[j] + h/lambda) - h/lambda
        
        #now we're going to look over all the slices of the gaussian
        for k = 1:2*ndeltas+1

            s = mu + deltas[k]

            if s <= xc[1]

                F[1,j] += ps[k];

            elseif s >= xc[n]

                F[n,j] += ps[k];

            else

                if xc[1] < s && xc[2] > s

                    lp,hp = 1,2;

                elseif xc[n-1] < s && xc[n] > s

                    lp,hp = n-1,n;

                else

                    hp,lp = ceil(Int, (s-xc[2])/dx) + 2, floor(Int, (s-xc[2])/dx) + 2;

                end

                if (hp == lp)

                    F[lp,j] += ps[k];

                else

                    dd = xc[hp] - xc[lp];
                    F[hp,j] += ps[k]*(s-xc[lp])/dd;
                    F[lp,j] += ps[k]*(xc[hp]-s)/dd;

                end

            end

        end

    end

end

function ll_wrapper_diffLR{TT}(py::Vector{TT}, data::Dict, beta::Vector{Float64}, mu0::Vector{Float64}, N::Int)
    
    #py[1:2*N] = exp.(py[1:2*N]);
    py[1:2*N] = 100 * 0.5*(1+tanh.(py[1:2*N]))
    py[2*N+1:4*N] = -10 + 20 * 0.5*(1+tanh.(py[2*N+1:4*N]))
    
    LL = LL_all_trials_diffLR(py, data, beta, mu0, N)
    
    return LL

end

function LL_all_trials_diffLR{TT}(py::Vector{TT}, data::Dict, beta::Vector{Float64}, mu0::Vector{Float64}, N::Int)
            
    py = reshape(py,N,dims["y"])

    output = pmap((L,R,T,N,SC) -> LL_single_trial_diffLR(py[N,:], L, R, T, spike_counts=SC),
        data["leftbups"], data["rightbups"], data["nT"], data["N"], data["spike_counts"])        

    py = vec(py)
    
    return -(sum(output) - sum(beta .* (py - mu0).^2))
    
end

function LL_single_trial_diffLR{TT}(py::Array{TT},
        L::Union{Array{Float64},Float64}, R::Union{Array{Float64},Float64}, T::Int;
        spike_counts::Union{Array{TT},Array{Int}}=Array{Int}(0,0))
    
    ΔLR = diffLR(T,L,R,path=true)
    lambda = my_sigmoid(ΔLR,py)
    LL = sum(spike_counts .*  log.(lambda*dt) - lambda * dt - lgamma.(spike_counts + 1))
    
    return LL
    
end

function diffLR(nT,L,R;path=false,dt::Float64=2e-2)
    
    #compute the cumulative diff of clicks
    t = 0:dt:nT*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    
    if path
        diffLR = cumsum(-L.weights + R.weights)
    else
        diffLR = sum(-L.weights + R.weights)
    end
    
end

end
