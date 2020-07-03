

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
    compute_gradient(pz, pd, data; n=53, state="state")
"""
function compute_gradient(pz::Dict{}, pd::Dict{}, data::Dict{};
    dx::Float64=0.1, state::String="state") where {TT <: Any}

    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx,state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_gradient(; ntrials=20000, n=53, dt=1e-2, use_bin_center=false, rng=1)
Generates default parameters, data and then computes the gradient
"""
function compute_gradient(; ntrials::Int=20000, dx::Float64=0.1,
        dt::Float64=1e-2, use_bin_center::Bool=false, rng::Int=1)

    pz, pd = default_parameters(generative=true)
    data = sample_clicks_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng)
    data = bin_clicks!(data,use_bin_center=use_bin_center, dt=dt)
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx, state="generative")
    ForwardDiff.gradient(ll, p_opt)

end


"""
    compute_Hessian(pz, pd, data; n=53, state="state")
"""
function compute_Hessian(pz::Dict{}, pd::Dict{}, data::Dict{};
    dx::Float64=0.1, state::String="state") where {TT <: Any}

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close(pz,pd,data; dx=dx,state=state)
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
function compute_CIs!(pz::Dict, pd::Dict, data::Dict; dx::Float64=0.1, state::String="final")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    lb = combine_latent_and_observation(pz["lb"], pd["lb"])
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])

    CI = Vector{Vector{Float64}}(undef,length(fit_vec))
    LLs = Vector{Vector{Float64}}(undef,length(fit_vec))
    xs = Vector{Vector{Float64}}(undef,length(fit_vec))

    ll_θ = compute_LL(pz[state], pd[state], data; dx=dx)

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





