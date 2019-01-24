#module initialize_spike_obs_model

#using Optim, LineSearches, StatsBase
#using latent_DDM_common_functions, ForwardDiff

#export ΔLR_ll_wrapper, filter_bound, filt_bound_model_wrapper

function filt_bound_model_wrapper(p_opt::Vector{TT},p_const::Vector{Float64}, fit_vec::BitArray{1},
        data::Dict,model_type::Union{String,Array{String}},f_str::String;
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),
        map_str::String = "exp", dt::Float64=1e-3, noise::String="Poisson") where {TT}
    
        #Wrapper function for computing LL of spike data from a latent variable with drift and bound
    
        #break up into latent and spiking parameters (located in global_functions)
        pz,py = latent_and_spike_params(p_opt, p_const, fit_vec, model_type, f_str)

        #map latent variables to bounded domain
        pz = map_latent_params!(pz,map_str,dt)   
        py = map(x->map_lambda_y_p!(x,f_str;map_str=map_str),py)
    
        #compute LL for each trial separately
        LL = pmap((T,L,R,N,k)->filt_bound_model(pz,py[N],T,L,R,k,f_str;dt=dt,map_str=map_str,noise=noise),
            data["nT"],data["leftbups"],data["rightbups"],data["N"],data["spike_counts"])
    
        #compute prior over spiking variables (located in latent variable, and somewhat redudant with a prior function within this module)
        LLprior = prior_on_spikes(py,mu0,beta) 
    
        #compute total LL
        LLprime = -(sum(LL) - LLprior)
    
        return LLprime
        
end

function filt_bound_model(pz::Vector{TT},
        py::Vector{Vector{TT}},
        T::Int,L::Vector{Float64},R::Vector{Float64},
        k::Union{Vector{Vector{Int}},Vector{Vector{Float64}}},f_str::String;
        dt::Float64=1e-3,map_str::String="exp",noise::String="Poisson") where {TT}
    
    #full latent and observation model for drift and bound latent
    
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
    
    λ = map(x->lambda_y(A,x,f_str),py);
    
    #if f_str == "exp"
        #py = map(x->map_exp_params!(x),py)
    #    λ = map(x->my_exp(A,x),py)
    #elseif f_str == "sig"
        #py = map(x->map_sig_params!(x,map_str),py)
    #    λ = map(x->my_sigmoid(A,x),py) 
    #end
    
    #compute the LL for each neuron on this trial, then sum across neurons
    LL = sum(map((λ,k)->sum(LL_func(λ,k,dt=dt,map_str=map_str,noise=noise)),λ,k))
        
end

function filter_bound(lambda::TT,B::TT,
        T::Int,L::Vector{Int},R::Vector{Int};dt::Float64=1e-3) where {TT}
    
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

################################### LATENT MODEL WITHOUT DRIFT OR BOUND ######################

function ΔLR_ll_wrapper(p::Vector{TT},
        k::Union{Vector{Vector{Int}},Vector{Vector{Float64}}}, ΔLR::Vector{Vector{Int}},f_str::String;
        map_str::String = "exp", dt::Float64=1e-3, 
        beta::Vector{Float64}=Vector{Float64}(), 
        mu0::Vector{Float64}=Vector{Float64}(),noise::String="Poisson",
        std0::Float64=1e-1) where {TT}
        
        #Wrapper function for computing LL of spike data from a latent variable with drift = 0 and no bound
        #Compute LL for each individual neuron separately
    
        #concatenate spike counts and $\int$ of $\Delta_{LR}$ across trials
        k = vcat(k...)
        ΔLR = vcat(ΔLR...)
    
        p = map_lambda_y_p!(p,f_str;map_str=map_str)
        λ = lambda_y(ΔLR,p,f_str);
    
        #compute LL for all trials, for this neuron
        LL = LL_func(λ,k,dt=dt,map_str=map_str,noise=noise,std0=std0)
    
        #include prior for single neuron (prior function is below, and somewhat redundant with a prior function that lives in the latent variable module
        LLprior = add_prior(p,mu0,beta)    
        LLprime = -(sum(LL) - LLprior)
            
end

#prior function designed to be used on individual neurons    
add_prior(p::Vector{TT},mu0::Vector{Float64},beta::Vector{Float64}) where {TT} = sum(map((b,p,mu0) -> b*(p - mu0)^2,beta,p,mu0))
    
function LL_func(λ::Vector{TT}, k::Union{Vector{Int},Vector{Float64}}; map_str::String = "exp", dt::Float64=1e-3, 
        noise::String="Poisson",std0::Float64=1e-1) where {TT}
    
    #loss function for single neuron      
    if noise == "Poisson"
        LL = poiss_likelihood(k,λ,dt) 
    elseif noise == "Gaussian"
        LL = gauss_likelihood(k,λ,std0=std0) 
    end
        
end

poiss_likelihood(k,λ,dt) = k.*log.(λ*dt) - λ*dt - lgamma.(k+1)

#noise is hardcoded in, this needs to be changed.
gauss_likelihood(k,λ;std0::Float64=1e-1) = log(1/sqrt(2*pi*std0^2)) - ((k-λ).^2)./(2*std0^2);
        
#end
