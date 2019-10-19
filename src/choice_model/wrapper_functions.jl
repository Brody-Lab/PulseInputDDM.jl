function compute_gradient(pz, pd, data; dx::Float64=0.25, state::String="state") where {TT <: Any}

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = ll_wrapper(x, data, parameter_map_f, dx=dx)

    return g = ForwardDiff.gradient(ll, p_opt)

end

function compute_Hessian(pz, pd, data; dx::Float64=0.25, state::String="state") where {TT <: Any}

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = ll_wrapper(x, data, parameter_map_f, dx=dx)

    return H = ForwardDiff.hessian(ll, p_opt);

end

function compute_H_CI!(pz, pd, data; dx::Float64=0.25)

    H = compute_Hessian(pz, pd, data; dx=dx)

    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["final"], pd["final"]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)

    return pz, pd, H

end


"""
    optimize_model(pz::Dict{}, pd::Dict{}, data::Dict{}; dx::Float64=0.25,
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true)

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Dict{}, pd::Dict{}, data::Dict{}; dx::Float64=0.25,
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true)

    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])

    check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    lb = combine_latent_and_observation(pz["lb"], pd["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])[fit_vec]
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["state"], pd["state"]), fit_vec)

    p_opt[p_opt .< lb] .= lb[p_opt .< lb]
    p_opt[p_opt .> ub] .= ub[p_opt .> ub]

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = ll_wrapper(x, data, parameter_map_f, dx=dx)
    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)

    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]

    return pz, pd, opt_output

end

function ll_wrapper(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        dx::Float64=0.25) where {TT <: Any}

    pz, pd = parameter_map_f(p_opt)

    LL = compute_LL(pz, pd, data; dx=dx)

    return -LL

end


"""
    compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any}

    compute LL for your model. returns a scalar

"""
compute_LL(pz::Vector{T}, pd::Vector{T}, data; dx::Float64=0.25) where {T <: Any} = sum(LL_all_trials(pz, pd, data, dx=dx))

split_latent_and_observation(p::Vector{TT}) where {TT} = p[1:dimz], p[dimz+1:end]

combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}}, 
    pd::Union{Vector{TT},BitArray{1}}) where {TT} = vcat(pz,pd)
