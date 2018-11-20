module poisson_neural_observation

using latent_DDM_common_functions, ForwardDiff, Optim, Pandas, Distributions

function do_H(p,fit_vec,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## compute Hessian
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    
    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CIz_plus, CIpy_plus = breakup(gather(p_opt + CI, p_const, fit_vec),f_str=f_str)
        map_pz!(CIz_plus,dt,map_str=map_str)
        map_py!.(CIpy_plus,f_str=f_str)

        CIz_minus, CIpy_minus = breakup(gather(p_opt - CI, p_const, fit_vec),f_str=f_str)
        map_pz!(CIz_minus,dt,map_str=map_str)
        map_py!.(CIpy_minus,f_str=f_str)
    
    else
        
        CIz_plus, CIz_minus = similar(pz),similar(pz)
        CIpy_plus, CIpy_minus = map(x->similar(x),deepcopy(py)),map(x->similar(x),deepcopy(py))
        
    end
    
    CIplus = vcat(CIz_plus,CIpy_plus) 
    CIminus = vcat(CIz_minus,CIpy_minus)    
    
    return CIplus, CIminus, H
    
end

function do_optim(p,fit_vec,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false)
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    inv_map_pz!(pz,dt,map_str=map_str)     
    inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, dt, n; 
        f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,
        show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)
    p = vcat(pz,py)   
        
    return p, opt_output, state
    
end

#function do_LL(p,fit_vec,dt,data,n::Int;
#        f_str="softplus",map_str::String="exp",
#        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
#        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())

function do_LL(p,dt,data,n::Int;
        f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    
    ###########################################################################################
    ## break up parameters based on which variables they relate to
    pz,py = breakup(p,f_str=f_str)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization
    #inv_map_pz!(pz,dt,map_str=map_str)     
    #inv_map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants
    #p_opt,p_const = inv_gather(inv_breakup(pz,py),fit_vec)
    
    ###########################################################################################
    ## Break up optimization vector into functional groups, remap to bounded domain and regroup
    #pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    #map_pz!(pz,dt,map_str=map_str)       
    #map_py!.(py,f_str=f_str)

    ###########################################################################################
    ## Compute LL
    #ll_wrapper(p_opt, p_const, fit_vec, data, dt, n; f_str=f_str, beta=beta, mu0=mu0, map_str=map_str)
    
    #11/5 changed this to deal with no priors, which is a default now
    #-(sum(LL_all_trials(pz, py, data, f_str=f_str, n=n, dt=dt)) - sum(gauss_prior.(py,mu0,beta)))
    LL = sum(LL_all_trials(pz, py, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

function compute_p0(ΔLR,k,dt;f_str::String="softplus",nconds::Int=7);
    
    #### compute linear regression slope of tuning to $\Delta_{LR}$ and miniumum firing based on binning and averaging

    conds_bins, = qcut(vcat(ΔLR...),nconds,labels=false,duplicates="drop",retbins=true)
    fr = map(i -> (1/dt)*mean(vcat(k...)[conds_bins .== i]),0:nconds-1)

    c = linreg(vcat(ΔLR...),vcat(k...))

    if f_str == "exp"
        p = vcat(minimum(fr),c[2])
    elseif f_str == "sig"
        p = vcat(minimum(fr),maximum(fr)-minimum(fr),c[2],0.)
    elseif f_str == "softplus"
        p = vcat(minimum(fr),c[2],0.)
    end
        
end

function do_p0(dt::Float64,data::Dict;f_str::String="softplus")
    
    ###########################################################################################
    ## Compute click difference and organize spikes by neuron
    ΔLR = pmap((T,L,R)->diffLR(T,L,R,data["dt"]),data["nT"],data["leftbups"],data["rightbups"])    
    trials, SC = group_by_neuron(data)
    
    pmap((trials,k)->compute_p0(ΔLR[trials],k,dt;f_str=f_str),trials,SC)
    
end
    
function do_optim_ΔLR(dt::Float64,data::Dict;
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        f_str::String="softplus",x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false)
    
        ###########################################################################################
        ## Compute p0 by linear regression
        p = do_p0(dt,data;f_str=f_str) 

        ###########################################################################################
        ## Compute click difference and organize spikes by neuron
        ΔLR = pmap((T,L,R)->diffLR(T,L,R,data["dt"]),data["nT"],data["leftbups"],data["rightbups"])    
        trials, SC = group_by_neuron(data)   
    
        ###########################################################################################
        ## Map parameters to unbounded domain for optimization
        inv_map_py!.(p,f_str=f_str)
    
        p = pmap((p,trials,k)->do_optim_ΔLR_single(p,dt,ΔLR[trials],k;show_trace=show_trace,
            f_str=f_str),p,trials,SC)
    
        ###########################################################################################
        ## Remap to bounded domain
        map_py!.(p,f_str=f_str)
    
end

function do_optim_ΔLR_single(p::Vector{Float64},dt::Float64,ΔLR::Vector{Vector{Int}},
        k::Vector{Vector{Int}};
        beta::Vector{Float64}=Vector{Float64}(),
        mu0::Vector{Float64}=Vector{Float64}(),f_str::String="softplus",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,iterations::Int=Int(5e3),
        show_trace::Bool=false)
    
        ###########################################################################################
        ## Optimize    
        ll(p) = ll_wrapper_ΔLR(p, k, ΔLR, dt; beta=beta, mu0=mu0, f_str=f_str)
        opt_output, state = opt_ll(p,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,show_trace=show_trace)
        p = Optim.minimizer(opt_output)
                
end

function ll_wrapper_ΔLR{TT}(p::Vector{TT},
        k::Vector{Vector{Int}}, ΔLR::Vector{Vector{Int}}, dt::Float64;
        beta::Vector{Float64}=Vector{Float64}(0),
        mu0::Vector{Float64}=Vector{Float64}(0),
        f_str::String="softplus")
       
        #check fy because of NaN poiss_LL fiasco
        map_py!(p,f_str=f_str)
        λ = fy.([p],vcat(ΔLR...),f_str=f_str)
    
        #-(sum(poiss_LL(λ,vcat(k...),dt)) - sum(gauss_prior(py,mu0,beta)))
        LL = sum(poiss_LL.(vcat(k...),λ,dt))
        length(beta) > 0 ? LL += sum(gauss_prior.(p,mu0,beta)) : nothing
    
        return -LL
            
end

function ll_wrapper{TT}(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, dt::Float64, n::Int; f_str::String="softplus", map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0))

    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)

    #11/5 changed this to deal with no priors, which is a default now
    #-(sum(LL_all_trials(pz, py, data, f_str=f_str, n=n, dt=dt)) - sum(gauss_prior.(py,mu0,beta)))
    LL = sum(LL_all_trials(pz, py, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return -LL
              
end
    
function LL_all_trials{TT}(pz::Vector{TT},py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}}, 
        data::Dict, dt::Float64, n::Int; f_str::String="softplus", comp_posterior::Bool=false)
        
    P,M,xc,dx, = P_M_xc(pz,n,dt)
    
    lambday = fy.(py,xc',f_str=f_str)'
    #lambday = reshape(vcat(lambday...),n,:);           
                
    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, lambday[:,N], SC, dt, n, comp_posterior=comp_posterior),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["N"],data["spike_counts"])        
    
end

function LL_single_trial{TT}(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
        comp_posterior::Bool=false)
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #spike count data
    spike_counts = reshape(vcat(spike_counts...),:,length(spike_counts))
    
    c = Vector{TT}(T)
    comp_posterior ? post = Array{Float64,2}(n,T) : nothing
    F = zeros(M) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = transition_Pa!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)        
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),1)));
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's   
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            P .*= vec(exp.(sum(poiss_LL.(spike_counts[t+1,:],lambday',dt),1)));           
            P,F = transition_Pa!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

function breakup(p; f_str::String="softplus")
                
    pz = p[1:dimz]

    if f_str == "sig"
        py = reshape(p[dimz+1:end],4,:)

    elseif f_str == "exp"
        py = reshape(p[dimz+1:end],2,:)

    elseif f_str == "softplus"
        py = reshape(p[dimz+1:end],3,:)

    end

    py = map(i->py[:,i],1:size(py,2))

    return pz, py
    
end

inv_breakup{TT}(pz::Vector{TT},py::Vector{Vector{TT}}) = vcat(pz,vcat(py...))

function sampled_dataset!(data::Dict, p::Vector{Float64}, dt::Float64; 
        f_str::String="softplus", dtMC::Float64=1e-4, num_reps::Int=1, rng::Int=1)

    construct_inputs!(data,num_reps)
    
    srand(rng)
    data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_model(p,T,L,R,N,dt;
            f_str=f_str, rng=rng), data["T"],data["leftbups"],data["rightbups"],
            data["N"], shuffle(1:length(data["N"])));        
    
    return data
    
end

function sample_model(p::Vector{Float64},T::Float64,L::Vector{Float64},R::Vector{Float64},
         N::Vector{Int}, dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1,
         ts::Float64=0.,get_fr::Bool=false)
    
    srand(rng)
    
    pz,py = breakup(p,f_str=f_str)
    A = sample_latent(T,L,R,pz;dt=dtMC)
    A = cat(1,zeros(Int(ts/dtMC)),A)
    A = decimate(A,Int(dt/dtMC))      
    
    #this is if only you want the spike counts of one cell, which happens when I sample the model to get fake data
    #could probably find a better way to do this
    if length(N) > 1 || get_fr == false
        Y = map(py -> poisson_noise.(fy.([py],A,f_str=f_str),dt),py[N])
    else
        Y = poisson_noise.(fy.(py[N],A,f_str=f_str),dt)
    end
    
end

function map_py!{TT}(p::Vector{TT};f_str::String="softplus",map_str::String="exp")
        
    if f_str == "exp"
        
        p[1] = exp(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = exp.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = 1e-5 + 99.99 * 0.5*(1+tanh.(p[1:2]))
            p[3:4] = -9.99 + 9.99*2 * 0.5*(1+tanh.(p[3:4]))
        end
        
    elseif f_str == "softplus"
          
        p[1] = exp(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function inv_map_py!{TT}(p::Vector{TT};f_str::String="softplus",map_str::String="exp")
     
    if f_str == "exp"

        p[1] = log(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = log.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = atanh.(((p[1:2] - 1e-5)/(99.99*0.5)) - 1)
            p[3:4] = atanh.(((p[3:4] + 9.99)/(9.99*2*0.5)) - 1)
        end
        
    elseif f_str == "softplus"
        
        p[1] = log(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function fy{TT}(p::Vector{TT},a::Union{TT,Float64,Int};f_str::String="softplus",x::Float64=0.,mu::Float64=0.,std::Float64=0.)
    
    if f_str == "sig"
        
        temp = p[3]*a + p[4]

        if exp(temp) < 1e-150
            y = p[1] + p[2]
        elseif exp(temp) >= 1e150
            y = p[1]
        else    
            y = p[1] + p[2]/(1. + exp(temp))
        end

        #protect from NaN gradient values
        #y[exp(temp) .<= 1e-150] = p[1] + p[2]
        #y[exp.(temp) .>= 1e150] = p[1]
        
    elseif f_str == "exp"
        
        y = p[1] + exp(p[2]*a)
        
    elseif f_str == "softplus"
            
        y = p[1] + log(1. + exp(p[2]*a + p[3]))
        
    elseif f_str == "expRBF"
            
        y = p[1] + exp(p[2]*a*pdf(Normal(mu,std),x))
        
    end
    
    return y
    
end

gauss_prior{TT}(p::Vector{TT}, mu::Vector{Float64}, beta::Vector{Float64}) = -sum(beta .* (p - mu).^2)

poiss_LL(k,λ,dt) = k*log(λ*dt) - λ*dt - lgamma(k+1)

poisson_noise(lambda,dt) = Int(rand(Poisson(lambda*dt)))

end
