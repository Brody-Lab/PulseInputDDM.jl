
#################################### Poisson neural observation model #########################

function compute_H_CI!(pz, py, data, trunc_μ_λ, dt)
    
    H = compute_Hessian(pz[:final],py[:final], pz[:fit], py[:fit], data; 
        λ0=trunc_μ_λ["by_trial"], f_str="softplus", dimy=3);

    #badindices = findall(abs.(vcat(pz[:final],vcat(py[:final]...))[vcat(pz[:fit],vcat(py[:fit]...))]) 
    #    .< 1e-4)

    #gooddims = setdiff(1:size(H,1),badindices)
    
    try
    
        gooddims = 1:size(H,1)

        evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
        otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
        gooddims = setdiff(gooddims,otherbad)

        fit_vec = pulse_input_DDM.combine_latent_and_observation(pz[:fit], py[:fit])
        p_opt, p_const = pulse_input_DDM.split_combine_invmap(deepcopy(pz[:final]), deepcopy(py[:final]), 
            fit_vec, dt, "softplus", "exp")

        CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));

        pz[:CI_plus], py[:CI_plus] = pulse_input_DDM.map_split_combine(p_opt + CI, p_const, fit_vec, 
            dt, "softplus", "exp", data["N0"], 3)
        pz[:CI_minus], py[:CI_minus] = pulse_input_DDM.map_split_combine(p_opt - CI, p_const, fit_vec, 
            dt,"softplus", "exp", data["N0"], 3)
        
    catch
        
        pz[:CI_plus] = typeof(pz[:final])
        pz[:CI_minus] = typeof(pz[:final])
        
        py[:CI_plus] = typeof(py[:final])
        py[:CI_minus] = typeof(py[:final])
        
    end
    
    return pz, py
    
end

function load_and_optimize(data, N; dt::Float64=1e-2, n::Int=53,
        fit_latent::BitArray{1}=trues(dimz), dimy::Int=4, show_trace::Bool=true,
        λ0::Dict=Dict(), f_str="sig", iterations::Int=Int(2e3), map_str::String="exp",
        fit_initial::Vector{Float64}=vcat(1e-6,15.,-0.1,10.,1.,0.2,0.005))
        
    #parameters of the latent model
    pz = DataFrames.DataFrame(name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        fit = fit_latent,
        initial = fit_initial);
    
    #parameters for the neural observation model
    py = DataFrames.DataFrame(name = repeat(["neuron"],outer=N),
        fit = repeat([trues(dimy)],outer=N),
        initial = repeat([Vector{Float64}(undef,dimy)],outer=N))
    
    py[:initial] = optimize_model(dt,data,py[:fit],show_trace=false,λ0=λ0["by_neuron"],f_str=f_str)
    
    pz[:final],py[:final],opt_output, state = optimize_model(pz[:initial],py[:initial],
        pz[:fit], py[:fit], data; λ0=λ0["by_trial"], f_str=f_str, show_trace=show_trace, n=n, dt=dt,
        iterations=iterations, dimy=dimy, map_str=map_str)
    
    return pz, py
    
end

function load_and_optimize(path::String, sessids, ratnames, N; dt::Float64=1e-2, n::Int=53,
        fit_latent::BitArray{1}=trues(dimz), dimy::Int=4, show_trace::Bool=true,
        λ0::Dict=Dict(), f_str="sig")
    
    data = make_data(path,sessids,ratnames,dt)
    
    #parameters of the latent model
    pz = DataFrames.DataFrame(name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        fit = fit_latent,
        initial = vcat(1e-6,15.,-0.1,10.,1.,0.2,0.005));
    
    #parameters for the neural observation model
    py = DataFrames.DataFrame(name = repeat(["neuron"],outer=N),
        fit = repeat([trues(dimy)],outer=N),
        initial = repeat([Vector{Float64}(undef,dimy)],outer=N))
    
    py[:initial] = optimize_model(dt,data,py[:fit],show_trace=false,λ0=λ0["by_neuron"],f_str=f_str)
    
    pz[:final],py[:final],opt_output, state = optimize_model(pz[:initial],py[:initial],
        pz[:fit], py[:fit], data; λ0=λ0["by_trial"], f_str=f_str, show_trace=show_trace, n=n, dt=dt)
    
    return pz, py
    
end

"""
    compute_Hessian(pz,py,pz_fit,py_fit,data;
        dt::Float64=1e-2, n::Int=53, f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())

    compute Hessian

"""
function compute_Hessian(pz,py,pz_fit,py_fit,data;
        dt::Float64=1e-2, n::Int=53, f_str="sig",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}},
        dimy::Int=4)
    
    N = length(py)
    fit_vec = combine_latent_and_observation(pz_fit,py_fit)
    p_opt, p_const = split_combine_invmap(pz, py, fit_vec, dt, f_str, map_str)
    
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, λ0, f_str, N; dt=dt, n=n, beta=beta, 
        mu0=mu0, map_str=map_str, dimy=dimy)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

"""
    optimize_model(pz,py,pz_fit,py_fit,data;
        dt::Float64=1e-2, n::Int=53, f_str="softplus",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=true, 
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Vector{TT}, py::Vector{Vector{TT}},data;
        dt::Float64=1e-2, n::Int=53, f_str="sig",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-4,
        iterations::Int=Int(2e3),show_trace::Bool=true, 
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}},
        dimy::Int=4) where {TT <: Any}
    
    #fix here
    check_pz!(pz)

    N = length(py)
    fit_vec = combine_latent_and_observation(pz["fit"],py["fit"])
    p_opt, p_const = split_combine_invmap(pz, py, fit_vec, dt, f_str, map_str)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, λ0, f_str, N; dt=dt, n=n, beta=beta, 
        mu0=mu0, map_str=map_str, dimy=dimy)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    pz, py = map_split_combine(p_opt, p_const, fit_vec, dt, f_str, map_str, N, dimy)
        
    return pz, py, opt_output, state
    
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, λ0::Vector{Vector{Vector{Float64}}}, f_str::String, N::Int;
        dt::Float64=1e-2, n::Int=53, map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),
        dimy::Int=4) where {TT}

    pz, py = map_split_combine(p_opt, p_const, fit_vec, dt, f_str, map_str, N, dimy)

    LL = compute_LL(pz, py, data, dt=dt, n=n, f_str=f_str, λ0=λ0, beta=beta, mu0 = mu0)
    
    return -LL
              
end

function compute_LL(pz::Vector{T},py::Vector{Vector{T}},data;
        dt::Float64=1e-2, n::Int=53,f_str="softplus",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}()) where {T <: Any}
    
    LL = sum(LL_all_trials(pz, py, data, dt=dt, n=n, f_str=f_str, λ0=λ0))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

######################### Deterministic latent variable model ####################################
    
function optimize_model(dt::Float64,data::Dict,fit_vec::Union{Vector{BitArray{1}},Vector{Vector{Bool}}};
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        f_str::String="softplus",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-4,
        iterations::Int=Int(1e1),show_trace::Bool=true,
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
    
        ###########################################################################################
        ## Compute click difference and organize spikes by neuron
        ΔLR = map((T,L,R)-> diffLR(T,L,R,data["dt"]), data["nT"], data["leftbups"], data["rightbups"])    
    
        p = map((trials,k)-> compute_p0(ΔLR[trials], k, dt; f_str=f_str), data["trial"], data["spike_counts_by_neuron"]) 
    
        inv_map_py!.(p,f_str=f_str)
    
        p = map((p,trials,k,fit_vec,λ0)-> optimize_model(p,dt,ΔLR[trials],k,fit_vec;
            show_trace=show_trace, f_str=f_str, λ0=λ0), p, data["trial"], data["spike_counts_by_neuron"], fit_vec, λ0)
    
        map_py!.(p,f_str=f_str)
    
end

function optimize_model(p::Vector{Float64},dt::Float64,ΔLR::Vector{Vector{Int}},
        k::Vector{Vector{Int}},fit_vec::Union{BitArray{1},Vector{Bool}};
        beta::Vector{Float64}=Vector{Float64}(),
        mu0::Vector{Float64}=Vector{Float64}(),f_str::String="softplus",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-4,iterations::Int=Int(1e1),
        show_trace::Bool=false,
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
        
        p_opt,p_const = split_variable_and_const(p,fit_vec)
    
        ###########################################################################################
        ## Optimize    
        ll(p_opt) = ll_wrapper(p_opt, p_const, fit_vec, k, ΔLR, dt; beta=beta, mu0=mu0, f_str=f_str, λ0=λ0)
        opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,show_trace=show_trace)
        p_opt = Optim.minimizer(opt_output)
    
        p = combine_variable_and_const(p_opt, p_const, fit_vec)
            
        return p
                
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}},
        k::Vector{Vector{Int}}, ΔLR::Vector{Vector{Int}}, dt::Float64;
        beta::Vector{Float64}=Vector{Float64}(0),
        mu0::Vector{Float64}=Vector{Float64}(0),
        f_str::String="softplus",
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT}
    
        py = combine_variable_and_const(p_opt, p_const, fit_vec)    
        map_py!(py,f_str=f_str)
    
        LL = compute_LL(py, ΔLR, k; dt=dt, f_str=f_str, beta=beta, mu0=mu0, λ0=λ0)
    
        return -LL
            
end

########################## Choice and neural model ###########################################

function compute_LL(pz::Vector{T}, py::Vector{Vector{T}}, bias::T, data::Dict;
        dt::Float64=1e-2, n::Int=53, f_str="softplus",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}()) where {T <: Any}
    
    LL = sum(LL_all_trials(pz, py, bias, data, dt=dt, n=n, f_str=f_str, λ0=λ0))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

########################## RBF model ###########################################

#=

function optimize_model(pz::Vector{TT}, py::Vector{Vector{TT}}, pRBF::Vector{Vector{TT}},
        pz_fit, py_fit, pRBF_fit, data;
        dt::Float64=1e-2, n::Int=53, f_str="sig2",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=false,
        dimy::Int=4, numRBF::Int=20) where {TT <: Any}

    N = length(py)
    fit_vec = combine_latent_and_observation(pz_fit, py_fit, pRBF_fit)    
    p_opt, p_const = split_combine_invmap(pz, py, pRBF, fit_vec, dt, f_str, map_str)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, f_str, N; dt=dt, n=n, beta=beta, 
        mu0=mu0, map_str=map_str, dimy=dimy, numRBF=numRBF)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)
    
    pz, py, pRBF = map_split_combine(p_opt, p_const, fit_vec, dt, f_str, map_str, N, dimy, numRBF)
        
    return pz, py, pRBF, opt_output, state
    
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, f_str::String, N::Int;
        dt::Float64=1e-2, n::Int=53, map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0),
        dimy::Int=4, numRBF::Int=20) where {TT}

    pz, py, pRBF = map_split_combine(p_opt, p_const, fit_vec, dt, f_str, map_str, N, dimy, numRBF)

    LL = compute_LL(pz, py, pRBF, data, dt=dt, n=n, f_str=f_str, beta=beta, mu0=mu0, numRBF=numRBF)
    
    return -LL
              
end

function compute_LL(pz::Vector{T},py::Vector{Vector{T}},pRBF::Vector{Vector{T}}, data;
        dt::Float64=1e-2, n::Int=53, f_str="sig2",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        numRBF::Int=20) where {T <: Any}
    
    LL = sum(LL_all_trials(pz, py, pRBF, data, dt=dt, n=n, f_str=f_str, numRBF=numRBF))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return LL
    
end

=#
