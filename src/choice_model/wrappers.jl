"""
    default_parameters(;generative=false)

Returns two dictionaries of default model parameters.
"""
function default_parameters(;generative::Bool=false)

    pd = Dict("name" => vcat("bias","lapse"),
              "fit" => vcat(false, false),
              "initial" => vcat(0.,0.),
              "lb" => [-30, 0.],
              "ub" => [30, 1.])

    #### NOOOOTE: a_prior = a/a+b, b_prior = a+b
    pz = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ","η","α_prior","β_prior","nd"],
              "fit" => vcat(false, true, false, false, true, false, false, true, true, false, true),
              "initial" => [eps(), 4. + rand()*(7. - 4.),-0.001, eps(), 2. + rand()* (4. - 2.), 0.15, 0.02, rand(), rand(), 2., rand()],
              "lb" => [0., 1., -5., 0., 0., 0.01, 0.005, 0., 0., 0., 0.],
              "ub" => [2., 15., 5., 100., 100., 1.2, 1., 1., 1., 30., 100.])

    if generative
        pz["generative"] = [eps(), 4. + rand()*(7. - 4.), rand()*2. - 1., rand()*2., rand()*2., rand()*1.5, rand(), rand(), rand(), 3. + rand()*(6. - 3.), rand()]
        pd["generative"] = [0.,0.0]
        pz["initial"][pz["fit"] .== 0] = pz["generative"][pz["fit"] .== 0]
    end

    return pz, pd

end


"""
    default_parameters_and_data(;generative=false,ntrials=2000,rng=1)
Returns default parameters and some simulated data
"""
function default_parameters_and_data(;generative::Bool=true, ntrials::Int=2000, rng::Int=1,
                                    dt::Float64=1e-4, use_bin_center::Bool=false)
    pz, pd = default_parameters(;generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center,dt=dt)

    return pz, pd, data

end


"""
    optimize_model(pz, pd; ntrials=20000, n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, dt=1e-2, use_bin_center=false, rng=1)

Generate data using known generative paramaeters (must be provided) and then optimize model
parameters using that data. Useful for testing the model fitting procedure.
"""
function optimize_model(pz::Dict{}, pd::Dict{}; ntrials::Int=20000, n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)

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
function optimize_model(; ntrials::Int=20000, n::Int=53,
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
function optimize_model(data::Dict{}; n::Int=53,
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
function optimize_model(pz::Dict{}, pd::Dict{}, data::Dict{}; n::Int=53,
        x_tol::Float64=1e-32, f_tol::Float64=1e-12, g_tol::Float64=1e-9,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    println("optimize! \n")
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])

    check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    lb = combine_latent_and_observation(pz["lb"], pd["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])[fit_vec]

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,pd,data; n=n, state="state")

    p_opt[p_opt .< lb] .= lb[p_opt .< lb]
    p_opt[p_opt .> ub] .= ub[p_opt .> ub]

    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, outer_iterations=outer_iterations, show_trace=show_trace)

    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    println("optimization complete \n")
    println("converged: $converged \n")

    return pz, pd, converged

end


"""
    compute_gradient(pz, pd, data; n=53, state="state")
"""
function compute_gradient(pz::Dict{}, pd::Dict{}, data::Dict{};
    n::Int=53, state::String="state") where {TT <: Any}

    p_opt, ll, = split_opt_params_and_close(pz,pd,data; n=n,state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_gradient(; ntrials=20000, n=53, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and then computes the gradient
"""
function compute_gradient(; ntrials::Int=20000, n::Int=53,
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
function compute_Hessian(pz::Dict{}, pd::Dict{}, data::Dict{};
    n::Int=53, state::String="state") where {TT <: Any}

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; n=n,state=state)
    ForwardDiff.hessian(ll, p_opt)

end


"""
    compute_CIs!(pz, pd, H)
"""
function compute_CIs!(pz::Dict, pd::Dict, H::Array{Float64,2})

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
function compute_CIs!(pz::Dict, pd::Dict, data::Dict; n::Int=53, state::String="final")

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
function LL_across_range(pz::Dict, pd::Dict, data::Dict, lb, ub; n::Int=53, state::String="state")

    nrange = 20
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    fit_vec_ids = findall(x-> x == 1, fit_vec)

    lb_vec = combine_latent_and_observation(lb[1][pz["fit"]], lb[2][pd["fit"]])
    ub_vec = combine_latent_and_observation(ub[1][pz["fit"]], ub[2][pd["fit"]])

    LLs = Array{Float64}(undef,length(lb_vec), nrange)
    xs = Array{Float64}(undef,length(lb_vec), nrange)

    ll_θ = compute_LL(pz[state], pd[state], data; n=n)

    for i = 1:length(lb_vec)

        println(i)

        fit_vec2 = falses(length(fit_vec))
        fit_vec2[fit_vec_ids[i]] = true

        p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec2)

        parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec2))
        ll(x) = compute_LL([x], data, parameter_map_f) - (ll_θ - 1.92)

        xs[i,:] = range(lb_vec[i], stop=ub_vec[i], length=nrange)
        LLs[i,:] = map(x->ll(x), xs[i,:])

    end

    return LLs, xs

end


"""
    LL_across_2params(pz, pd, data)

"""
function LL_across_2params(pz::Dict, pd::Dict, data::Dict, lb, ub, prm; nrange = 10, n::Int=53, state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])

    lb_vec = combine_latent_and_observation(lb[1], lb[2])
    ub_vec = combine_latent_and_observation(ub[1], ub[2])

    LLs = Array{Float64}(undef,nrange*nrange)
    xs = Array{Float64}(undef,length(fit_vec), nrange)
    ys = Array{Float64}(undef,length(fit_vec), nrange)
    mesh = Array{Float64,2}(undef,nrange*nrange ,2)

    fit_vec2 = falses(length(fit_vec))
    fit_vec2[prm[1]] = true
    fit_vec2[prm[2]] = true

    ll_θ = compute_LL(pz[state], pd[state], data; n=n)

    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec2)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec2))
    ll(x) = compute_LL(x, data, parameter_map_f) - (ll_θ - 1.92)

    xs = range(lb_vec[prm[1]], stop=ub_vec[prm[1]], length=nrange)
    ys = range(lb_vec[prm[2]], stop=ub_vec[prm[2]], length=nrange)
    i = 1
    for x = xs, y = ys
        print(i)
        mesh[i,1] = x
        mesh[i,2] = y
        LLs[i] = ll([x,y])
        i = i + 1
    end
    

    return LLs, mesh

end


"""
    compute_LL(x, data, parameter_map_f; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function compute_LL(x::Vector{TT}, data::Dict, parameter_map_f::Function; n::Int=53) where {TT <: Any}

    pz, pd = parameter_map_f(x)
    compute_LL(pz, pd, data; n=n)

end


"""
    compute_LL(pz, pd, data; n=53)

Computes the log likelihood of the animal's choices (data["pokedR"] in data) given the model parameters
contained within the Vectors pz and pd.
"""
compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials(pz, pd, data, n=n))


"""
    compute_LL(pz, pd, data; n=53, state="state")

Computes the log likelihood of the animal's choices (data["pokedR"] in data) given the model parameters
contained within the Dicts pz and pd. The optional argument `state` determines which key
(e.g. initial, final, state, generative, etc.) will be used (since the functions
this function calls accepts Vectors of Floats)
"""
function compute_LL(pz::Dict{}, pd::Dict{}, data::Dict{}; n::Int=53, state::String="state") where {T <: Any}
    compute_LL(pz[state], pd[state], data, n=n)
end


"""
    compute_LL(; ntrials=2e4, n=53, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and computes the LL of that data
"""
function compute_LL(; ntrials::Int=20000, n::Int=53,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    sum(LL_all_trials(pz["generative"], pd["generative"], data, n=n))

end


"""
"""
function split_opt_params_and_close(pz::Dict{}, pd::Dict{}, data::Dict{}; n::Int=53, state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = -compute_LL(x, data, parameter_map_f, n=n)

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
