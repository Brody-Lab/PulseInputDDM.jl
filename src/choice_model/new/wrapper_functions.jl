function compute_Hessian_new(pz, pd, data; n::Int=53) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(pz["final"], pd["final"], fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_new(x, data, parameter_map_f, n=n)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end


function compute_H_CI_new!(pz, pd, data; n::Int=53)
    
    H = compute_Hessian_new(pz, pd, data; n=n)
    
    gooddims = 1:size(H,1)
    
    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(copy(pz["final"]), copy(pd["final"]), 
        fit_vec, data["dt"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))
    
    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #identify which ML parameters have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(pd, "generative")
        pd["within_CI"] = (pd["CI_minus"] .< pd["generative"]) .& (pd["CI_plus"] .> pd["generative"])
    end
    
    return pz, pd
    
end


function optimize_model_new(pz::Dict{}, pd::Dict{}, data; n=53,
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
        
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])
    
    pz = check_pz!(pz)
            
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(pz["state"], pd["state"], fit_vec, data["dt"], pz["lb"], pz["ub"])

    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_new(x, data, parameter_map_f, n=n)
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    
    return pz, pd, opt_output, state
        
end

function ll_wrapper_new(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        n::Int=53) where {TT <: Any}
          
    pz, pd = parameter_map_f(p_opt)
    
    LL = compute_LL_new(pz, pd, data; n=n)
    
    return -LL
              
end

compute_LL_new(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials_new(pz, pd, data, n=n))
