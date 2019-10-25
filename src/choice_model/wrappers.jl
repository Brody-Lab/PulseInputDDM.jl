"""
    default_parameters(;generative=false)

Returns two dictionaries of default model parameters.
"""
function default_parameters(;generative::Bool=false)

    pd = Dict("name" => vcat("bias","lapse"),
              "fit" => vcat(true, true),
              "initial" => vcat(0.5,0.1),
              "lb" => [-Inf, 0.],
              "ub" => [Inf, 1.])

    pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
              "fit" => vcat(false, true, true, true, true, true, true),
              "initial" => [eps(), 10., -0.1, 20., 0.5, 0.8, 0.008],
              "lb" => [0., 8., -5., 0., 0., 0.01, 0.005],
              "ub" => [2., 40., 5., 100., 2.5, 1.2, 1.])

    if generative
        pz["generative"] = [eps(), 10., -0.1, 20., 0.5, 0.8, 0.008]
        pd["generative"] = [0.5,0.1]
    end

    return pz, pd

end


"""
    optimize_model(; ntrials=20000, dx:=0.25, x_tol=1e-12, f_tol=1e-12, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, dt=1e-2, use_bin_center=false, rng=1)

Generate data using known generative paramaeters and then optimize model
parameters using that data. Useful for testing the model fitting procedure.
"""
function optimize_model(; ntrials::Int=20000, dx::Float64=0.25,
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)

    pz, pd, converged = optimize_model(pz, pd, data; dx=dx,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, iterations=iterations, show_trace=show_trace)

    return pz, pd, converged

end


"""
    optimize_model(data; dx=0.25, x_tol=1e-12, f_tol=1e-12, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)

Optimize model parameters using default parameter initialization.
"""
function optimize_model(data::Dict{}; dx::Float64=0.25,
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true)

    pz, pd = default_parameters()
    pz, pd, converged = optimize_model(pz, pd, data; dx=dx,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol,
        iterations=iterations, show_trace=show_trace)

    return pz, pd, converged

end


"""
    optimize_model(pz, pd, data; dx=0.25, x_tol=1e-12, f_tol=1e-12, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)

Optimize model parameters. pz and pd are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
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

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,pd,data; dx=dx,state=state)

    p_opt[p_opt .< lb] .= lb[p_opt .< lb]
    p_opt[p_opt .> ub] .= ub[p_opt .> ub]

    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace)

    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]

    return pz, pd, converged

end


"""
    compute_gradient(pz, pd, data; dx=0.25, state="state")
"""
function compute_gradient(pz::Dict{}, pd::Dict{}, data::Dict{};
    dx::Float64=0.25, state::String="state") where {TT <: Any}

    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx,state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_gradient(; ntrials=20000, dx=0.25, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and then computes the gradient
"""
function compute_gradient(; ntrials::Int=20000, dx::Float64=0.25,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx, state="generative")
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_Hessian(pz, pd, data; dx=0.25, state="state")
"""
function compute_Hessian(pz::Dict{}, pd::Dict{}, data::Dict{};
    dx::Float64=0.25, state::String="state") where {TT <: Any}

    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx,state=state)
    ForwardDiff.hessian(ll, p_opt);

end


"""
    compute_CIs!(pz, pd, data)
"""
function compute_CIs!(pz, pd, H)

    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,pd,Dict{}; state="final")

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)

    return pz, pd

end


"""
    ll_wrapper(p_opt, data, parameter_map_f; dx=0.25)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input,
and compute the negative log likelihood of the data given the parametes. Used
in optimization.
"""
function ll_wrapper(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        dx::Float64=0.25) where {TT <: Any}

    pz, pd = parameter_map_f(p_opt)
    -compute_LL(pz, pd, data; dx=dx)

end


"""
    compute_LL(pz, pd, data; dx=0.25)

Computes the log likelihood of the animal's choices (data["pokedR"] in data) given the model parameters
contained within the Vectors pz and pd.
"""
compute_LL(pz::Vector{T}, pd::Vector{T}, data; dx::Float64=0.25) where {T <: Any} = sum(LL_all_trials(pz, pd, data, dx=dx))


"""
    compute_LL(pz, pd, data; dx=0.25, state="state")

Computes the log likelihood of the animal's choices (data["pokedR"] in data) given the model parameters
contained within the Dicts pz and pd. The optional argument `state` determines which key
(e.g. initial, final, state, generative, etc.) will be used (since the functions
this function calls accepts Vectors of Floats)
"""
function compute_LL(pz::Dict{}, pd::Dict{}, data::Dict{}; dx::Float64=0.25, state::String="state") where {T <: Any}
    sum(LL_all_trials(pz[state], pd[state], data, dx=dx))
end


"""
    compute_LL(; ntrials=2e4, dx=0.25, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and computes the LL of that data
"""
function compute_LL(; ntrials::Int=20000, dx::Float64=0.25,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    sum(LL_all_trials(pz["generative"], pd["generative"], data, dx=dx))

end


"""
"""
function split_opt_params_and_close(pz::Dict{}, pd::Dict{}, data::Dict{}; dx::Float64=0.25, state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = ll_wrapper(x, data, parameter_map_f, dx=dx)

    return p_opt, ll, parameter_map_f

end


"""
    split_latent_and_observation(p)

Splits a vector up into two vectors. The first vector is for components related
to the latent variables, the second is for components related to the observation model.
### Examples
```jldoctest
julia> pz, pd = pulse_input_DDM.default_parameters();

julia> p = pulse_input_DDM.combine_latent_and_observation(pz["initial"], pd["initial"]);

julia> pulse_input_DDM.split_latent_and_observation(p) == (pz["initial"], pd["initial"])
true
```
"""
split_latent_and_observation(p::Vector{TT}) where {TT} = p[1:dimz], p[dimz+1:end]


"""
    combine_latent_and_observation(pz,pd)

Combines two vectors into one vector. The first vector is for components related
to the latent variables, the second vectoris for components related to the observation model.
### Examples
```jldoctest
julia> pz, pd = pulse_input_DDM.default_parameters();

julia> p = pulse_input_DDM.combine_latent_and_observation(pz["initial"], pd["initial"]);

julia> pulse_input_DDM.split_latent_and_observation(p) == (pz["initial"], pd["initial"])
true
```
"""
combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}},
    pd::Union{Vector{TT},BitArray{1}}) where {TT} = vcat(pz,pd)
