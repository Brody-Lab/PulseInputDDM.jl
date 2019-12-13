## create struct
## unpacking
## mapping
## jacobians and hessian
## add logpdf and rand for LL and sample
"""
    default_parameters_and_data(;generative=false,ntrials=2000,rng=1)
Returns default parameters and some simulated data
"""
function default_model(; ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, centered::Bool=false)

    θ = θchoice()
    clicks, choices = rand(θ, ntrials; rng=rng)
    binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)

    return θ, choicedata(binned_clicks, choices)

end


"""
    optimize_model(pz, pd; ntrials=20000, n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, dt=1e-2, use_bin_center=false, rng=1)

Generate data using known generative paramaeters (must be provided) and then optimize model
parameters using that data. Useful for testing the model fitting procedure.
"""
function optimize(pz::Dict{}, pd::Dict{}; ntrials::Int=20000, n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks(data,use_bin_center=use_bin_center, dt=dt)

    pz, pd, converged = optimize_model(pz, pd, data; n=n,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, iterations=iterations, show_trace=show_trace)

    return pz, pd, data, converged

end


"""
    optimize_model(; ntrials=20000, n=53, x_tol=1e-10, f_tol=1e-16, g_tol=1e-3,
        iterations=Int(2e3), show_trace=tru dt=1e-2, use_bin_center=false, rng=1,
        outer_iterations=Int(1e1))

Generate data using known generative paramaeters and then optimize model
parameters using that data. Useful for testing the model fitting procedure.
"""
function optimize(; ntrials::Int=20000, n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1,
        outer_iterations::Int=Int(1e1))

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)

    pz, pd, converged = optimize_model(pz, pd, data; n=n,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, iterations=iterations,
        show_trace=show_trace, outer_iterations=outer_iterations)

    return pz, pd, data, converged

end


"""
    optimize_model(data; n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, outer_iterations=Int(1e1))

Optimize model parameters using default parameter initialization.
"""
function optimize(data::Dict{}; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    pz, pd = default_parameters()
    pz, pd, converged = optimize_model(pz, pd, data; n=n,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol,
        iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    return pz, pd, converged

end


"""
    optimize_model(pz, pd, data; n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, outer_iterations=Int(1e1))

Optimize model parameters. pz and pd are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.

BACK IN THE DAY TOLS WERE: x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

"""
function optimize(data::choicedata, options::opt; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    println("optimize! \n")

    @unpack binned_clicks, choices = data
    @unpack fit, lb, ub, x0 = options

    F = as(Tuple(as.(Real, lb, ub)))

    x0 = collect(inverse(F, Tuple(x0)))
    c, x0 = x0[.!fit], x0[fit]
    #x_c(x) = x_c(x,c,fit)
    ℓℓ(y) = -loglikelihood(y, binned_clicks, choices; n=n)
    Fℓℓ(x) = ℓℓ(collect(transform(F, x_c(x,c,fit))))
    #Fℓℓ(x) = transform_logdensity(F, ℓℓ, x_c(x))

    output = opt_func(x0, Fℓℓ; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace)

    x, converged = Optim.minimizer(output), Optim.converged(output)
    x = collect(transform(F, x_c(x,c,fit)))

    θ = pack!(x)
    model = choiceDDM(θ, binned_clicks, choices)

    println("optimization complete. converged: $converged \n")

    return model, converged

end


"""
    loglikelihood(x, model; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{TT}, binned_clicks, choices; n::Int=53) where {TT <: Real}

    θ = pack!(x)
    sum(loglikelihood(θ, binned_clicks, choices; n=n))

end


"""
    loglikelihood(; ntrials=2e4, n=53, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and computes the LL of that data
"""
function loglikelihood(; ntrials::Int=20000, n::Int=53,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    sum(LL_all_trials(pz["generative"], pd["generative"], data, n=n))

end


"""
    gradient(pz, pd, data; n=53, state="state")
"""
function gradient(pz::Dict{}, pd::Dict{}, data::Dict{};
    n::Int=53, state::String="state") where {TT <: Any}

    p_opt, ll, = split_opt_params_and_close(pz,pd,data; n=n,state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_gradient(; ntrials=20000, n=53, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and then computes the gradient
"""
function gradient(; ntrials::Int=20000, n::Int=53,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; n=n, state="generative")
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_Hessian(pz, pd, data; n=53, state="state")
"""
function Hessian(pz::Dict{}, pd::Dict{}, data::Dict{};
    n::Int=53, state::String="state") where {TT <: Any}

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; n=n,state=state)
    ForwardDiff.hessian(ll, p_opt)

end


"""
    compute_CIs!(pz, pd, H)
"""
function CIs(pz::Dict, pd::Dict, H::Array{Float64,2})

    println("computing confidence intervals \n")

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8)

    try
        gooddims = 1:size(H,1)
        evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
        otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
        gooddims = setdiff(gooddims,otherbad)
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))
    catch
        @warn "CI computation failed."
    end

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,pd,Dict(); state="final")

    pz["CI_plus_hessian"], pd["CI_plus_hessian"] = parameter_map_f(p_opt + CI)
    pz["CI_minus_hessian"], pd["CI_minus_hessian"] = parameter_map_f(p_opt - CI)

    return pz, pd

end


"""
    compute_CIs!(pz, pd, data)

Computes confidence intervals based on the likelihood ratio test
"""
function CIs(pz::Dict, pd::Dict, data::Dict; n::Int=53, state::String="final")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    lb = combine_latent_and_observation(pz["lb"], pd["lb"])
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])

    CI = Vector{Vector{Float64}}(undef,length(fit_vec))
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

        xs[i] = range(lb[i], stop=ub[i], length=50)
        LLs[i] = map(x->ll(x), xs[i])
        idxs = findall(diff(sign.(LLs[i])) .!= 0)

        #CI[i] = sort(find_zeros(ll, lb[i], ub[i]; naive=true, no_pts=3))

        CI[i] = []

        for j = 1:length(idxs)
            newroot = find_zero(ll, (xs[i][idxs[j]], xs[i][idxs[j]+1]), Bisection())
            push!(CI[i], newroot)
        end

        if length(CI[i]) > 2
            @warn "More than three roots found. Uh oh."
        end

        if length(CI[i]) == 0
            CI[i] = vcat(lb[i], ub[i])
        end

        if length(CI[i]) == 1
            if CI[i][1] < p_opt[1]
                CI[i] = sort(vcat(CI[i], ub[i]))
            elseif CI[i][1] > p_opt[1]
                CI[i] = sort(vcat(CI[i], lb[i]))
            end
        end

    end

    try
        pz["CI_plus_LRtest"], pd["CI_plus_LRtest"] = split_latent_and_observation(map(ci-> ci[2], CI))
        pd["CI_minus_LRtest"], pd["CI_minus_LRtest"] = split_latent_and_observation(map(ci-> ci[1], CI))
    catch
        @warn "something went wrong putting CI into pz and pd"
    end

    return pz, pd, CI, LLs, xs

end


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


"""
    pack(p)

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
function pack!(x::Vector{TT}) where {TT <: Real}

    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse = x
    θ = θchoice(θz(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ), bias,lapse)

end
