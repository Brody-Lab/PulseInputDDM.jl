module initialize_spike_obs_model

const dimy = 4

using Optim, LineSearches, StatsBase, latent_model
using global_functions, ForwardDiff

export x0_spikes, opt_func, x0_ll_wrapper, fit_and_CV, do_ML_filt_bound, filter_bound

function do_ML_filt_bound(p_z,p_y,fit_vec,model_type,map_str,dt,data,beta_y,mu0_y)
    
    ###########################################################################################
    ## Map parameters to unbounded domain for optimization

    p_z = inv_map_latent_params!(p_z,map_str,dt)  
    p_y = map(x->inv_map_sig_params!(x,map_str),p_y);

    ###########################################################################################
    ## Concatenate into a single vector and break up into optimization variables and constants

    p = vcat(p_z,vcat(p_y...))

    p_opt = p[fit_vec]
    p_const = p[.!fit_vec];

    ###########################################################################################
    ## Optimize

    ll(x) = filt_bound_model_wrapper(x, p_const, fit_vec, data, model_type,
        beta_y=beta_y, mu0_y=mu0_y, dt=dt);
    
    od = OnceDifferentiable(ll, p_opt; autodiff=:forward);
    
    p_opt = Optim.minimizer(Optim.optimize(od, p_opt, 
                BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
                linesearch = BackTracking()), 
                Optim.Options(g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                iterations = 1000, store_trace = true, 
                show_trace = true, extended_trace = false, allow_f_increases = true)));

    ###########################################################################################
    ## Break up optimization vector into functional groups and remap to bounded domain

    p_z,p_y = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)
    p_z = map_latent_params!(p_z,map_str,dt)   
    p_y = map(x->map_sig_params!(x,map_str),p_y);
    
    ###########################################################################################
    ## compute Hessian

    H = ForwardDiff.hessian(ll, p_opt);
    d,V = eig(H)
    
    if all(d .> 0)
    
        CI = 2*sqrt.(diag(inv(H)));
    
        CI_z_plus, CI_y_plus = latent_and_spike_params(p_opt + CI, p_const, fit_vec, model_type)
        CI_z_plus = map_latent_params!(CI_z_plus,map_str,dt) 
        CI_y_plus = map((x)->map_sig_params!(x,map_str), CI_y_plus)

        CI_z_minus, CI_y_minus = latent_and_spike_params(p_opt - CI, p_const, fit_vec, model_type)
        CI_z_minus = map_latent_params!(CI_z_minus,map_str,dt)
        CI_y_minus = map((x)->map_sig_params!(x,map_str),CI_y_minus);
    else
        
        CI_z_plus = similar(p_z);
        CI_z_minus = similar(p_z);
        
    end
    
    return p_z, CI_z_plus, CI_z_minus
    
end

function filt_bound_model_wrapper{TT}(p_opt::Vector{TT},p_const::Vector{Float64}, fit_vec::BitArray{1},
        data::Dict,model_type::Union{String,Array{String}};
        beta_y::Vector{Float64}=Vector{Float64}(0), 
        mu0_y::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),
        map_str::String = "exp", dt::Float64=1e-3)
    
        pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type)

        pz = map_latent_params!(pz,map_str,dt)   
        py = map(x->map_sig_params!(x,map_str),py)
    
        #compute LL for each trial separately
        LL = pmap((T,L,R,N,k)->filt_bound_model(pz,py[N],T,L,R,k;dt=dt,map_str=map_str),
            data["nT"],data["leftbups"],data["rightbups"],data["N"],data["spike_counts"])
    
        LLprior = prior_on_spikes(py,mu0_y,beta_y) 
    
        #compute total LL
        LLprime = -(sum(LL) - LLprior)
    
        return LLprime
        
end

function filt_bound_model{TT}(pz::Vector{TT},
        py::Vector{Vector{TT}},
        T::Int,L::Vector{Float64},R::Vector{Float64},
        k::Vector{Vector{Int}};dt::Float64=1e-3,map_str::String="exp")
    
    #break up latent parameters
    B,lambda_drift = pz[3:4];
        
    #bin the clicks
    t = 0:dt:T*dt;
    L = fit(Histogram,L,t,closed=:left)
    R = fit(Histogram,R,t,closed=:left)
    L = L.weights;
    R = R.weights;
      
    #compute the latent variable
    A = filter_bound(lambda_drift,B,T,L,R,dt=dt)
    
    #compute the LL for each neuron on this trial, then sum across neurons
    LL = sum(map((py,k)->sum(LL_func(py,k,A,dt=dt,map_str=map_str)),py,k))
    
    return LL
    
end

function filter_bound{TT}(lambda::TT,B::TT,
        T::Int,L::Vector{Int},R::Vector{Int};dt::Float64=1e-3)
    
    #for saving the latent trajectory on a trial
    A = Vector{TT}(T)
    a = zero(TT)

    @inbounds for t = 1:T 
        
            #jump and drift
            a += (dt*lambda) * a + (-L[t] + R[t])

            #hit bound?
            abs(a) > B ? (a = B * sign(a); A[t:T] = a; break) : A[t] = a   
        
    end
    
    return A
    
end

function fit_and_CV(py0,k,trials,ΔLR,ntrials,mu0,beta0;map_str="exp",dt=1e-3,rng=1)
    
    idx_train = sort(sample(srand(rng),1:length(k),ntrials,replace=false))
    
    if isempty(setdiff(1:length(k),idx_train))
        idx_test = idx_train
    else
        idx_test = setdiff(1:length(k),idx_train)
    end

    k_train = k[idx_train]
    trials_train = trials[idx_train]

    k_test = k[idx_test]
    trials_test = trials[idx_test]
    
    beta_y = vcat(beta0*ones(4))
    
    py = opt_func(py0,k_train,trials_train,ΔLR,map_str=map_str,dt=dt,beta=beta_y,mu0=mu0)
    
    #be careful of this, note that when this wrapper function is not used in the context of an optimization
    # the input py is mutuable.
    LL_CV = x0_ll_wrapper(copy(py),k_test,ΔLR[trials_test],map_str=map_str,dt=dt,beta=beta_y,mu0=mu0)
    
    py = map_sig_params!(py,map_str)
    
    return LL_CV, py
    
end

function x0_spikes(py0::Vector{Vector{Float64}},data,map_str::String,
        beta::Vector{Float64},mu0::Vector{Vector{Float64}};
        dt::Float64=1e-3)
    
    ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=dt),data["nT"],data["leftbups"],data["rightbups"]);
    
    #make sigmoid parameters
    #py0 = cat(1,log.(1e-2),log.(1e-2), zeros(2));
    
    #optimize all neurons
    pystar = pmap((py0,k,trials,mu0)->opt_func(py0,k,trials,ΔLR,map_str=map_str,dt=dt,
            beta=beta,mu0=mu0),py0,data["spike_counts"],data["trial"],mu0);
    
    #map pystar parameters
    map(x->map_sig_params!(x,map_str),pystar)
    
    return pystar
    
end

function opt_func(p0,k,trials,ΔLR;map_str::String="exp",dt::Float64=1e-3,
    beta::Vector{Float64}=Vector{Float64}(dimy), mu0::Vector{Float64}=Vector{Float64}(dimy))
    
    Optim.minimizer(optimize(p0 -> x0_ll_wrapper(p0,k,ΔLR[trials],map_str=map_str,dt=dt,beta=beta,mu0=mu0), 
        p0, method = Optim.BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), 
        linesearch = BackTracking()), autodiff=:forward, g_tol = 1e-12, x_tol = 1e-16, 
        f_tol = 1e-16, iterations = Int(1e16), show_trace = false, allow_f_increases = true));
    
end

function add_prior{TT}(p::Vector{TT},mu0::Vector{Float64},beta::Vector{Float64})
    
        LL_prior = sum(map((b,p,mu0) -> b*(p - mu0)^2,beta,p,mu0))
    
end

function x0_ll_wrapper{TT}(p::Vector{TT},
        k::Vector{Vector{Int}}, ΔLR::Vector{Vector{Int}};
        map_str::String = "exp", dt::Float64=1e-3, 
        beta::Vector{Float64}=Vector{Float64}(dimy), 
        mu0::Vector{Float64}=Vector{Float64}(dimy))
    
        map_sig_params!(p,map_str)
    
        k = vcat(k...)
        ΔLR = vcat(ΔLR...)
        LL = LL_func(p,k,ΔLR,dt=dt,map_str=map_str)
    
        LLprior = add_prior(p,mu0,beta)    
        LLprime = -(sum(LL) - LLprior)
    
        return LLprime
        
end

function LL_func{TT}(p::Vector{TT}, k::Vector{Int}, x::Union{Vector{Int},Vector{TT}}; 
        map_str::String = "exp", dt::Float64=1e-3)
        
    λ = my_sigmoid(x,p);      
    LL = poiss_likelihood(k,λ,dt) 
    
    return LL
    
end

function poiss_likelihood(k,λ,dt)
    
    LL = k.*log.(λ*dt) - λ*dt - lgamma.(k+1)
    
    return LL
    
end

end
