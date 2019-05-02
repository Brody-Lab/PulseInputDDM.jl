
#################################### Poisson neural observation model #########################

function compute_H_CI!(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String)
    
    H = compute_Hessian(pz, py, data, f_str);

    #badindices = findall(abs.(vcat(pz[:final],vcat(py[:final]...))[vcat(pz[:fit],vcat(py[:fit]...))]) 
    #    .< 1e-4)

    #gooddims = setdiff(1:size(H,1),badindices)
        
    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));

    pz["CI_plus"], py["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], py["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #if generative parameters exist, identify which ones have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_bounds"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(py, "generative")
        py["within_bounds"] = map((x,y,z)-> [((x .< z) .& (y .> z)) for (x,y,z) in zip(x,y,z)], 
            py["CI_minus"], py["CI_plus"], py["generative"])
    end
    
    return pz, py
    
end

"""
    compute_Hessian(pz,py,data;n::Int=53, f_str="softplus")

    compute Hessian

"""
function compute_Hessian(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String; 
        n::Int=53)
    
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str; n=n)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

function load_and_optimize(path::String, sessids, ratnames, f_str; n::Int=53, 
        dt::Float64=1e-2, delay::Float64=0.,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3))
    
    data = aggregate_spiking_data(path,sessids,ratnames)
    data = bin_clicks_spikes_and_λ0!(data; dt=dt, delay=delay)
    
    pz, py = load_and_optimize(data, f_str; n=n,
        pz=pz, show_trace=show_trace, iterations=iterations)
    
    return pz, py
    
end

function load_and_optimize(data::Vector{Dict{Any,Any}}, f_str; n::Int=53,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3))
    
    nsessions = length(data)
    N_per_sess = map(data-> data["N"], data)
    
    if f_str == "softplus"
        dimy = 3
    end
    
    #parameters for the neural observation model
    py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), N_per_sess),
        "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in N_per_sess],
        "dimy"=> dimy,
        "N"=> N_per_sess,
        "nsessions"=> nsessions)
    
    py["initial"] = map(data -> optimize_model(data,f_str,show_trace=false), data)
    
    pz, py, = optimize_model(pz, py, data, f_str; 
        show_trace=show_trace, n=n, iterations=iterations)
    
    return pz, py
    
end

"""
    optimize_model(pz,py,data;
        n::Int=53, f_str="softplus"
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=true)

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        n::Int=53, x_tol::Float64=1e-4, f_tol::Float64=1e-6, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true) where {TT <: Any}
    
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])
    
    pz = check_pz!(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["state"], py["state"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])

    ###########################################################################################
    ## Optimize
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str; n=n)
    
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
        
    return pz, py, opt_output, state
    
end

function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String; 
        n::Int=53) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)   
    LL = compute_LL(pz, py, data; n=n, f_str=f_str)
        
    return -LL
              
end

function compute_LL(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}};
        n::Int=53, f_str="softplus") where {T <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials(pz, py, data, n=n, f_str=f_str)), py, data))
            
end

######################### Deterministic latent variable model (only observation noise) ############
    
function optimize_model(data::Dict, f_str::String;
        x_tol::Float64=1e-4, f_tol::Float64=1e-6, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
    
        ###########################################################################################
        ## Compute click difference and organize spikes by neuron
        ΔLR = map((T,L,R)-> diffLR(T,L,R, data["dt"]), data["nT"], data["leftbups"], data["rightbups"])   
        SC_byneuron = group_by_neuron(data["spike_counts"], data["ntrials"], data["N"])
        λ0_byneuron = group_by_neuron(data["λ0"], data["ntrials"], data["N"])
    
        py = map(k-> compute_p0(ΔLR, k, data["dt"], f_str), SC_byneuron) 
    
        inv_map_py!.(py, f_str=f_str)
    
        p = pmap((py,k,λ0)-> optimize_model(py, ΔLR, k, data["dt"], f_str, λ0; show_trace=show_trace), 
            py, SC_byneuron, λ0_byneuron)
    
        map_py!.(py, f_str=f_str)
    
end

function optimize_model(py::Vector{Float64}, ΔLR::Vector{Vector{Int}}, k::Vector{Vector{Int}}, 
        dt::Float64, f_str::String, λ0::Vector{Vector{Float64}}; 
        x_tol::Float64=1e-4, f_tol::Float64=1e-6, g_tol::Float64=1e-2, iterations::Int=Int(2e3),
        show_trace::Bool=false)
            
        ###########################################################################################
        ## Optimize    
        ll(x) = ll_wrapper(x, ΔLR, k, dt, f_str, λ0)
        opt_output, state = opt_ll(py, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
            iterations=iterations, show_trace=show_trace)
        py = Optim.minimizer(opt_output)
                
        return py
                
end

function ll_wrapper(py::Vector{TT}, ΔLR::Vector{Vector{Int}}, k::Vector{Vector{Int}}, 
        dt::Float64, f_str::String, λ0::Vector{Vector{Float64}}) where {TT <: Any}
    
        map_py!(py,f_str=f_str)   
        -compute_LL(py, ΔLR, k, dt, f_str, λ0)
                
end

function compute_LL(py::Vector{TT}, ΔLR::Vector{Vector{Int}}, k::Vector{Vector{Int}},
        dt::Float64, f_str::String, λ0::Vector{Vector{Float64}}) where {TT <: Any}
    
    sum(poiss_LL.(vcat(k...), fy22(py, vcat(ΔLR...), vcat(λ0...), f_str=f_str), dt))
           
end

function compute_p0(ΔLR,k,dt,f_str;nconds::Int=7)
    
    conds_bins, = qcut(vcat(ΔLR...),nconds,labels=false,duplicates="drop",retbins=true)
    fr = map(i -> (1/dt)*mean(vcat(k...)[conds_bins .== i]),0:nconds-1)

    A = vcat(ΔLR...)
    b = vcat(k...)
    c = hcat(ones(size(A, 1)), A) \ b

    if f_str == "exp"
        p = vcat(minimum(fr),c[2])
    elseif (f_str == "sig") | (f_str == "sig2")
        p = vcat(minimum(fr),maximum(fr)-minimum(fr),c[2],0.)
    elseif f_str == "softplus"
        p = vcat(minimum(fr),c[2],0.)
    end
        
end

#=

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

    
function compute_LL_and_prior(pz::Vector{T}, py::Vector{Vector{T}}, λ0::Vector{Vector{Vector{Float64}}}, data;
        n::Int=53, f_str="softplus",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {T <: Any}
    
    LL = sum(map((py,data,λ0)-> sum(LL_all_trials(pz, py, λ0, data, n=n, f_str=f_str))))
        
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
        
    return LL
           
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, λ0::Vector{Vector{Vector{Float64}}}, f_str::String;
        n::Int=53, beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0)) where {TT}

    #stopped here
    pz, py = map_split_combine(p_opt, p_const, fit_vec, data[1]["dt"], f_str, N, dimy)
    
    LL = compute_LL_and_prior(pz, py, λ0, data; n=n, f_str=f_str)
        
    return -LL
              
end

=#
