#################################### Full Poisson neural observation model #############
"""
    optimize_model(pz,py,data;
        n::Int=53, f_str="softplus"
        x_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-12,
        iterations::Int=Int(5e3),show_trace::Bool=true)

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        n::Int; x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
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
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
        
    return pz, py, opt_output, state
    
end


function optimize_model_con(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        n::Int; x_tol::Float64=1e-16, f_tol::Float64=1e-16, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e3)) where {TT <: Any}
    
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])
    
    pz = check_pz!(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["state"], py["state"]),fit_vec)
    
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    opt_output = opt_ll_con(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace,outer_iterations=outer_iterations);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
        
    return pz, py, opt_output
    
end

function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String, 
        n::Int) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)   
    LL = compute_LL(pz, py, data, n, f_str)
        
    return -LL
              
end

function compute_LL(pz::Vector{T}, py::Vector{Vector{Vector{U}}}, data::Vector{Dict{Any,Any}},
        n::Int, f_str::String) where {T,U <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials(pz, py, data, n, f_str)), py, data))
            
end


############ Compute Hessian ##############

function compute_H_CI!(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, n::Int)
    
    H = compute_Hessian(pz, py, data, f_str, n)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    
    try               
        gooddims = 1:size(H,1)
        evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
        otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
        gooddims = setdiff(gooddims,otherbad)
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));
    catch
        @warn "CI computation failed."
    end

    pz["CI_plus"], py["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], py["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #if generative parameters exist, identify which ones have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(py, "generative")
        py["within_CI"] = map((x,y,z)-> [((x .< z) .& (y .> z)) for (x,y,z) in zip(x,y,z)], 
            py["CI_minus"], py["CI_plus"], py["generative"])
    end
    
    return pz, py, H
    
end

function compute_H_CI_con!(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, n::Int)
    
    H = compute_Hessian_con(pz, py, data, f_str, n)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["final"], py["final"]),fit_vec)
    
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    
    try               
        gooddims = 1:size(H,1)
        evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
        otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
        gooddims = setdiff(gooddims,otherbad)
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));
    catch
        @warn "CI computation failed."
    end

    pz["CI_plus"], py["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], py["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #if generative parameters exist, identify which ones have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(py, "generative")
        py["within_CI"] = map((x,y,z)-> [((x .< z) .& (y .> z)) for (x,y,z) in zip(x,y,z)], 
            py["CI_minus"], py["CI_plus"], py["generative"])
    end
    
    return pz, py, H
    
end



"""
    compute_Hessian(pz,py,data;n::Int=53, f_str="softplus")

    compute Hessian

"""
function compute_Hessian(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
    
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

function compute_Hessian_con(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
     
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["final"], py["final"]),fit_vec)
    
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

function compute_grad_con(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        n::Int)
     
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["state"], py["state"]),fit_vec)
    
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)
    
    return g = ForwardDiff.gradient(ll, p_opt)
        
end

#=

######### Deterministic latent variable model with lambda=0 (only observation noise) ############
    
function optimize_model(data::Dict, f_str::String;
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
    
        ###########################################################################################
        ## Compute click difference and organize spikes by neuron
        ΔLR = map((T,L,R)-> diffLR(T,L,R, data["dt"]), data["nT"], data["leftbups"], data["rightbups"])   
        SC_byneuron = group_by_neuron(data["spike_counts"], data["ntrials"], data["N"])
        λ0_byneuron = group_by_neuron(data["λ0"], data["ntrials"], data["N"])
    
        py = map(k-> compute_p0(ΔLR, k, data["dt"], f_str), SC_byneuron) 
    
        inv_map_py!.(py, f_str=f_str)
    
        py = map((py,k,λ0)-> optimize_model(py, ΔLR, k, data["dt"], f_str, λ0; show_trace=show_trace), 
            py, SC_byneuron, λ0_byneuron)
    
        map_py!.(py, f_str=f_str)
    
end

function optimize_model(py::Vector{Float64}, ΔLR::Vector{Vector{Int}}, k::Vector{Vector{Int}}, 
        dt::Float64, f_str::String, λ0::Vector{Vector{Float64}}; 
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2, iterations::Int=Int(2e3),
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
    
    sum(pmap((ΔLR,k,λ0)-> sum(poiss_LL.(k, map((ΔLR, λ0)-> f_py(ΔLR, λ0, py, f_str=f_str), ΔLR, λ0), dt)), ΔLR,k,λ0))   
           
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

########################## Model w prior ###########################################
    
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

########### testing functions

function compute_LL_threads(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}},
        n::Int, f_str::String) where {T <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials_threads(pz, py, data, n, f_str=f_str)), py, data))
            
end

=#
