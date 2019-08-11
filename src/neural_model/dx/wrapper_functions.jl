#################################### dx Poisson neural observation model #############

function compute_H_CI_dx!(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, dx::Float64)
    
    H = compute_Hessian_dx(pz, py, data, f_str, dx)
        
    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    
    try
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
    
    return pz, py
    
end

function compute_Hessian_dx(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String, 
        dx::Float64)
    
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_dx(x, data, parameter_map_f, f_str, dx)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

function optimize_model_dx(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        dx::Float64; x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
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
    ll(x) = ll_wrapper_dx(x, data, parameter_map_f, f_str, dx)
    
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
        
    return pz, py, opt_output, state
    
end

function ll_wrapper_dx(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String, 
        dx::Float64) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)   
    LL = compute_LL_dx(pz, py, data, dx, f_str)
        
    return -LL
              
end

function compute_LL_dx(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}},
        dx::Float64, f_str::String) where {T <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials_dx(pz, py, data, dx, f_str=f_str)), py, data))
            
end
