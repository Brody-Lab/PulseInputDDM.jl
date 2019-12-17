"""
    mean_exp_rate_per_trial(pz, py, data, f_str; use_bin_center=false, dt=1e-2, num_samples=100)
Given parameters and model inputs returns the average expected firing rate of the model computed over num_samples number of samples.
"""
function mean_exp_rate_per_trial(pz, py, data, f_str::String; use_bin_center::Bool=false, dt::Float64=1e-2,
        num_trials::Int=100)

    output = map(i-> sample_expected_rates_multiple_sessions(pz, py, data, f_str, use_bin_center, dt; rng=i), 1:num_samples)
    mean(map(k-> output[k][1], 1:length(output)))

end



"""
    mean_exp_rate_per_cond(pz, py, data, f_str; use_bin_center=false, dt=1e-2, num_samples=100)

"""
function mean_exp_rate_per_cond(pz, py, data, f_str::String; use_bin_center::Bool=false, dt::Float64=1e-2,
        num_trials::Int=100)

    μ_rate = mean_exp_rate_per_trial(pz, py, data, f_str; use_bin_center=use_bin_center, dt=dt, num_samples=num_samples)

    map(i-> condition_mean_varying_duration_trials(μ_rate[i], data[i]["conds"],
            data[i]["nconds"], data[i]["N"], data[i]["nT"]), 1:length(data))

end


"""
"""
function condition_mean_varying_duration_trials(μ_rate, conds, nconds, N, nT)

    map(n-> map(c-> [mean([μ_rate[conds .== c][k][n][t]
        for k in findall(nT[conds .== c] .>= t)])
        for t in 1:(maximum(nT[conds .== c]))],
                1:nconds), 1:N)

end


"""
"""
function boot_LL(pz,py,data,f_str,i,n)
    dcopy = deepcopy(data)
    dcopy["spike_counts"] = sample_spikes_multiple_sessions(pz, py, [dcopy], f_str; rng=i)[1]

    LL_ML = compute_LL(pz, py, [dcopy], n, f_str)

    #LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n->
    #            neural_null(d["spike_counts"][r][n], d["λ0"][r][n], d["dt"]),
    #                +, 1:d["N"]), +, 1:d["ntrials"]), +, [data])

    #(LL_ML - LL_null) / dcopy["ntrials"]

    LL_null = mapreduce(d-> mapreduce(r-> mapreduce(n->
        neural_null(d["spike_counts"][r][n], map(λ-> f_py(0.,λ, py[1][n],f_str), d["λ0"][r][n]), d["dt"]),
            +, 1:d["N"]), +, 1:d["ntrials"]), +, [dcopy])

    #return 1. - (LL_ML/LL_null), LL_ML, LL_null
    LL_ML - LL_null

end


"""
"""
function sample_clicks_and_spikes(Ψ,
        num_sessions::Int, num_trials_per_session::Vector{Int}; centered::Bool=false,
        dt::Float64=1e-4, rng::Int=0)

    @unpack θy,θz = Ψ
    @unpack θ,f = θy

    clicks = map((ntrials,rng)-> synthetic_clicks(ntrials; rng=rng), num_trials_per_session, (1:num_sessions) .+ rng)
    λ0 = map((clicks,θ) -> sample_λ0(clicks.T, θ; dt=dt), clicks, θ)

    Y = sample_spikes_multiple_sessions(θz, θ, clicks, λ0, f, centered, dt; rng=rng)

    return clicks, λ0, Y

end

function sample_λ0(T, py::Vector{Vector{Float64}}; dt::Float64=1e-4, rng::Int=1)

    #data["dt_synthetic"], data["synthetic"], data["N"] = dtMC, true, length(py)

    #Random.seed!(rng)
    #data["λ0"] = [repeat([collect(range(10. *rand(),stop=10. * rand(),
    #                    length=Int(ceil(T./dt))))], outer=length(py)) for T in data["T"]]
    λ0 = [repeat([zeros(Int(ceil(T./dt)))], outer=length(py)) for T in T]

end


"""
"""
function sample_spikes_multiple_sessions(θz::θz, py::Vector{Vector{Vector{Float64}}},
        clicks, λ0, f_str::String, centered::Bool, dt::Float64; rng::Int=1)

    λ, = sample_expected_rates_multiple_sessions(θz, py, clicks, λ0, f_str, centered, dt; rng=rng)
    Y = map(λ-> map(λ-> map(λ-> poisson_noise!.(λ, dt), λ), λ), λ)

end


"""
"""
function sample_expected_rates_multiple_sessions(θz::θz, py::Vector{Vector{Vector{Float64}}},
        clicks, λ0, f_str::String, centered::Bool, dt::Float64; rng::Int=1)

    nsessions = length(clicks)

    output = map((clicks, λ0, py)-> sample_expected_rates_single_session(clicks, λ0, θz, py, f_str, centered, dt; rng=rng),
        clicks, λ0, py)

    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)

    return λ, a

end


"""
"""
function sample_expected_rates_single_session(clicks, λ0, θz::θz, py::Vector{Vector{Float64}},
        f_str::String, centered::Bool, dt::Float64; rng::Int=1)

    #@unpack clicks, λ0 = inputs
    @unpack T,L,R,ntrials = clicks
    Random.seed!(rng)
    #rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)
    @unpack nT, nL, nR = binned_clicks

    output = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(θz,py,λ0,nT,L,R,nL,nR,
        f_str,centered,dt; rng=rng), λ0, nT, L, R, nL, nR, shuffle(1:ntrials))

    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)

    return λ,a

end


"""
"""
function sample_expected_rates_single_trial(θz::θz, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}},
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int},
        f_str::String, centered::Bool, dt::Float64; rng::Int=1)

    Random.seed!(rng)
    a = rand(θz,nT,L,R,nL,nR; centered=centered, dt=dt)
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), py, λ0)

    return λ, a

end


"""
"""
poisson_noise!(lambda,dt) = Int(rand(Poisson(lambda*dt)))
