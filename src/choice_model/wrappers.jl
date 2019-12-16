"""
"""
function pack(x::Vector{TT}) where {TT <: Real}

    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse = x
    θ = θchoice(θz(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ), bias,lapse)

end


"""
    unpack(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function unpack(θ::θchoice)

    @unpack θz, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    x = collect((σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse))

    return x

end


"""
    optimize_model(data; options=opt(), n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)

Optimize model parameters. data is a struct that contains the binned clicks and the choices.
options is a struct that containts the initial values, boundaries,
and specification of which parameters to fit.

BACK IN THE DAY TOLS WERE: x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

"""
function optimize(data::choicedata; options::opt=opt(), n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, x0 = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data; n=n)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = pack(x)
    model = choiceDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model, options

end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{TT}, data; n::Int=53) where {TT <: Real}

    θ = pack(x)
    loglikelihood(θ, data; n=n)

end



"""
    loglikelihood(choiceDDM; n=53)

Computes the log likelihood for a set of trials consistent with the animal's choice on each trial.
```
"""
function loglikelihood(model::choiceDDM; n::Int=53)

    @unpack θ, data = model
    loglikelihood(θ, data; n=n)

end


"""
    gradient(model; options, n=53)
"""
function gradient(model::choiceDDM; n::Int=53)

    @unpack θ, data = model
    x = unpack(θ)
    ℓℓ(x) = -loglikelihood(x, data; n=n)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model; options, n=53)
"""
function Hessian(model::choiceDDM; n::Int=53)

    @unpack θ, data = model
    x = unpack(θ)
    ℓℓ(x) = -loglikelihood(x, data; n=n)

    ForwardDiff.hessian(ℓℓ, x)

end


"""
    CIs(H)
"""
function CIs(model::choiceDDM, H::Array{Float64,2})

    @unpack θ = model
    HPSD = Matrix(cholesky(Positive, H, Val{false}))

    if !isapprox(HPSD,H)
        @warn "Hessian is not positive definite. Approximated by closest PSD matrix."
    end

<<<<<<< HEAD
    try
        pz["CI_plus_LRtest"], pd["CI_plus_LRtest"] = split_latent_and_observation(map(ci-> ci[2], CI))
        pd["CI_plus_LRtest"], pd["CI_minus_LRtest"] = split_latent_and_observation(map(ci-> ci[1], CI))
    catch
        @warn "something went wrong putting CI into pz and pd"
    end

    return pz, pd, CI, LLs, xs
=======
    CI = 2*sqrt.(diag(inv(HPSD)))
>>>>>>> constrained_newAPI

end


#=
"""
    LL_across_range(pz, pd, data)

"""
function LL_across_range(pz::Dict, pd::Dict, data::Dict, lb, ub; n::Int=53, state::String="final")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])

    lb_vec = combine_latent_and_observation(lb[1], lb[2])
    ub_vec = combine_latent_and_observation(ub[1], ub[2])

    LLs = Vector{Vector{Float64}}(undef,length(fit_vec))
    xs = Vector{Vector{Float64}}(undef,length(fit_vec))

    ll_θ = compute_LL(pz[state], pd[state], data; n=n)

    for i = 1:length(fit_vec)

        println(i)

        fit_vec2 = falses(length(fit_vec))
        fit_vec2[i] = true

        p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec2)

        parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec2))
        ll(x) = compute_LL([x], data, parameter_map_f) - (ll_θ - 1.92)

        xs[i] = range(lb_vec[i], stop=ub_vec[i], length=50)
        LLs[i] = map(x->ll(x), xs[i])

    end

    return LLs, xs

end

<<<<<<< HEAD

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
    compute_LL(pz[state], pd[state], data, dx=dx)
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
=======
=#
>>>>>>> constrained_newAPI
