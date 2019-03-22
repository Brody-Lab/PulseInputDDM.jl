function splitdata_train_test(data, trunc_μ_λ; frac::Float64=0.8, dt::Float64=1e-2)
    
    divided_data, divided_λ = train_test_divide(data, trunc_μ_λ, frac)
    
    pz, py = load_and_optimize(divided_data["train"], divided_data["train"]["N0"]; 
        fit_latent= vcat(falses(1),trues(4),falses(2)), 
        show_trace= false, λ0=divided_λ["train"], f_str="softplus",dimy=3)
    
    LL_stim = compute_LL(pz[:final], py[:final], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    #LL_null = compute_LL(pz[:final], [vcat(py[:final][1][1],0.,py[:final][1][3])], 
    #        divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    LL_null = neural_null(vcat(vcat(divided_data["test"]["spike_counts"]...)...),
        vcat(vcat(divided_λ["test"]["by_trial"]...)...), dt) 

    LL_stim - LL_null
    
end

function splitdata_train_test_multi(data, trunc_μ_λ; frac::Float64=0.9)
    
    divided_data, divided_λ = train_test_divide_multi(data ,trunc_μ_λ, frac)
    
    pz, py = load_and_optimize(divided_data["train"], divided_data["train"]["N0"]; 
        fit_latent= vcat(falses(1),trues(4),falses(2)), 
        show_trace= false, λ0=divided_λ["train"], f_str="softplus", dimy = 3)
    
    LL_stim = compute_LL(pz[:final], py[:final], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    LL_null = compute_LL(pz[:final], [vcat(py[:final][1][1],0.,py[:final][1][3])], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")

    LL_stim - LL_null
    
end




function opt_ll(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16,iterations::Int=Int(5e3),
        show_trace::Bool=true)
    
    obj = OnceDifferentiable(ll, p_opt; autodiff=:forward)
    m = BFGS(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())
    options = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = iterations, store_trace = true, show_trace = show_trace, 
        extended_trace = false, allow_f_increases = true)
    lbfgsstate = Optim.initial_state(m, options, obj, p_opt)
    
    output = Optim.optimize(obj, p_opt, m, options, lbfgsstate)
    
    return output, lbfgsstate
    
end

opt_ll_Newton(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16) = 
        Optim.minimizer(Optim.optimize(OnceDifferentiable(ll, p_opt; autodiff=:forward), 
        p_opt,Newton(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
        linesearch = BackTracking()), Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = Int(5e3), store_trace = true, show_trace = true, 
        extended_trace = false, allow_f_increases = true)))

#################################### Choice observation model #################################

function load_and_optimize(path::String, sessids, ratnames; dt::Float64=1e-2, n::Int=53,
        fit_latent::BitArray{1}=trues(dimz), show_trace::Bool=true,
        map_str::String="exp")
    
    data = make_data(path,sessids,ratnames,dt)
    
    #parameters of the latent model
    pz = DataFrames.DataFrame(name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        fit = fit_latent,
        initial = vcat(1e-6,20.,-0.1,100.,5.,1.,0.08));
    
    #parameters for the choice observation
    pd = DataFrames.DataFrame(name = "bias", fit = true, initial = 0.);
    
    pz[:final], pd[:final], = optimize_model(pz[:initial], pd[:initial][1], 
        pz[:fit], pd[:fit], data; dt=dt, n=n, show_trace=show_trace, map_str=map_str)
    
    return pz, pd
    
end

function load_and_optimize(data; dt::Float64=1e-2, n::Int=53,
        fit_latent::BitArray{1}=trues(dimz), show_trace::Bool=true,
        map_str::String="exp")
        
    #parameters of the latent model
    pz = DataFrames.DataFrame(name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        fit = fit_latent,
        initial = vcat(1e-6,20.,-0.1,100.,5.,1.,0.08));
    
    #parameters for the choice observation
    pd = DataFrames.DataFrame(name = "bias", fit = true, initial = 0.);
    
    pz[:final], pd[:final], = optimize_model(pz[:initial], pd[:initial][1], 
        pz[:fit], pd[:fit], data; dt=dt, n=n, show_trace=show_trace, map_str=map_str)
    
    return pz, pd
    
end

function compute_Hessian(pz::Vector{TT},bias::TT,pz_fit_vec,bias_fit_vec,data;
        dt::Float64=1e-2,n::Int=53,map_str::String="exp") where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz_fit_vec, bias_fit_vec)
    p_opt, p_const = split_combine_invmap(pz, bias, fit_vec, dt, map_str)
    
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, n=n, dt=dt, map_str=map_str)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end

function compute_CI(H, pz::Vector{Float64}, bias::Float64, pz_fit_vec, bias_fit_vec, data;
        dt::Float64=1e-2,n::Int=53,map_str::String="exp")
    
    fit_vec = combine_latent_and_observation(pz_fit_vec, bias_fit_vec)
    p_opt, p_const = split_combine_invmap(pz, bias, fit_vec, dt, map_str)
    
    CI = 2*sqrt.(diag(inv(H)))
    
    CIz_plus, CIbias_plus = map_split_combine(p_opt + CI, p_const, fit_vec, dt, map_str)
    CIz_minus, CIbias_minus = map_split_combine(p_opt - CI, p_const, fit_vec, dt, map_str)
    
    return CIz_plus, CIbias_plus, CIz_minus, CIbias_minus
    
end

"""
    optimize_model(pz, bias, pz_fit_vec, bias_fit_vec,
        data; dt, n, map_str, x_tol,f_tol,g_tol, iterations)

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Vector{TT}, bias::TT, pz_fit_vec, bias_fit_vec,
        data; dt::Float64=1e-2, n=53, map_str::String="exp",
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-4,
        iterations::Int=Int(2e3),show_trace::Bool=true) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz_fit_vec, bias_fit_vec)
    p_opt, p_const = split_combine_invmap(pz, bias, fit_vec, dt, map_str)

    ###########################################################################################
    ## Optimize
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, n=n, dt=dt, map_str=map_str)
    opt_output, state = opt_ll(p_opt,ll;g_tol=g_tol,x_tol=x_tol,f_tol=f_tol,iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz, bias = map_split_combine(p_opt, p_const, fit_vec, dt, map_str)
    
    return pz, bias, opt_output, state
        
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, 
        fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict; n::Int=53, dt::Float64=1e-2, map_str::String="exp") where {TT}
          
    pz, bias = map_split_combine(p_opt, p_const, fit_vec, dt, map_str)
    
    #lapse is like adding a prior out here?
    #modified 11/8 lapse needs to be dealt with different not clear that old way was correct
    #P *= (1.-inatt)
    #P += inatt/n
    
    LL = compute_LL(pz,bias,data;dt=dt,n=n)
    
    return -LL
              
end

"""
    compute_LL(pz,bias,data;dt::Float64=1e-2,n::Int=53)

    compute LL for choice observation model

"""
compute_LL(pz::Vector{T},bias::T,data;dt::Float64=1e-2,n::Int=53) where {T <: Any} = 
    sum(LL_all_trials(pz, bias, data, n=n, dt=dt))

#################################### Poisson neural observation model #########################

function load_and_optimize(data, N; dt::Float64=1e-2, n::Int=53,
        fit_latent::BitArray{1}=trues(dimz), dimy::Int=4, show_trace::Bool=true,
        λ0::Dict=Dict(), f_str="sig", iterations::Int=Int(2e3), map_str::String="exp")
        
    #parameters of the latent model
    pz = DataFrames.DataFrame(name = vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        fit = fit_latent,
        initial = vcat(1e-6,15.,-0.1,10.,1.,1.,0.08));
    
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
        initial = vcat(1e-6,15.,-0.1,10.,1.,1.,0.08));
    
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

function compute_CI(H, pz::Vector{Float64}, py::Vector{Vector{Float64}}, 
        pz_fit_vec, py_fit, data;
        dt::Float64=1e-2, n::Int=53, f_str="sig", map_str::String="exp",
        dimy::Int=4)
    
    N = length(py)
    fit_vec = combine_latent_and_observation(pz_fit,py_fit)
    p_opt, p_const = split_combine_invmap(pz, py, fit_vec, dt, f_str, map_str)
    
    CI = 2*sqrt.(diag(inv(H)))
    
    CIz_plus, CIbias_plus = map_split_combine(p_opt + CI, p_const, fit_vec, dt, map_str, N, dimy)
    CIz_minus, CIbias_minus = map_split_combine(p_opt - CI, p_const, fit_vec, dt, map_str, N, dimy)
    
    return CIz_plus, CIbias_plus, CIz_minus, CIbias_minus
    
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
function optimize_model(pz::Vector{TT},py::Vector{Vector{TT}},pz_fit,py_fit,data;
        dt::Float64=1e-2, n::Int=53, f_str="sig",map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-4,
        iterations::Int=Int(2e3),show_trace::Bool=true, 
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}},
        dimy::Int=4) where {TT <: Any}

    N = length(py)
    fit_vec = combine_latent_and_observation(pz_fit,py_fit)
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

########################## Priors ###########################################

gauss_prior(p::Vector{TT}, mu::Vector{Float64}, beta::Vector{Float64}) where {TT} = -sum(beta .* (p - mu).^2)
