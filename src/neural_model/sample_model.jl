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
function synthetic_data(θ::θneural,
        nsess::Int, ntrials::Vector{Int}, ncells; centered::Bool=true,
        dt::Float64=1e-2, rng::Int=0, dt_synthetic=1e-4)

    spikes,clicks,λ0 = rand(θ, nsess, ntrials, ncells; rng=rng)

    spikes = map(y-> bin_spikes(y, dt; dt_synthetic=dt_synthetic, synthetic=true), spikes)
    λ0 = map(λ0-> bin_λ0(λ0, dt; synthetic=true), λ0)
    binned_clicks = map(clicks-> bin_clicks.(clicks, centered=centered, dt=dt), clicks)

    input_data = map((clicks, binned_clicks, λ0)-> neuralinputs.(clicks, binned_clicks, λ0, dt, centered),
        clicks, binned_clicks, λ0)

    map((input_data,spikes,ncells) -> neuraldata.(input_data,spikes,ncells),
        input_data, spikes, ncells)

end

function synthetic_λ0(clicks, N; dt::Float64=1e-4, rng::Int=1)

    @unpack T = clicks
    #data["dt_synthetic"], data["synthetic"], data["N"] = dtMC, true, length(py)

    #Random.seed!(rng)
    #data["λ0"] = [repeat([collect(range(10. *rand(),stop=10. * rand(),
    #                    length=Int(ceil(T./dt))))], outer=length(py)) for T in data["T"]]
    #λ0 = [repeat([zeros(Int(ceil(T./dt)))], outer=N) for T in T]
    λ0 = repeat([zeros(Int(ceil(T/dt)))], outer=N)

end


"""
"""
function rand(θ::θneural, nsess, ntrials, ncells; centered::Bool=false, dt::Float64=1e-4, rng::Int=1)

    @unpack θy,θz,f = θ

    clicks = synthetic_clicks.(ntrials, collect((1:nsess) .+ rng))
    binned_clicks = map(clicks-> bin_clicks.(clicks, centered=centered, dt=dt), clicks)

    λ0 = map((clicks,ncells) -> synthetic_λ0.(clicks, ncells; dt=dt), clicks, ncells)

    input_data = map((clicks, binned_clicks, λ0)-> neuralinputs.(clicks, binned_clicks, λ0, dt, centered),
        clicks, binned_clicks, λ0)

    output = map((input_data, θy)-> rand(θz, θy, f, input_data; rng=rng),
        input_data, θy)

    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)
    Y = map(λ-> map(λ-> map(λ-> poisson_noise!.(λ, dt), λ), λ), λ)

    return Y, clicks, λ0

end


"""
"""
function rand(θz::θz, θy::Vector{Vector{Float64}},
        f_str::String, input_data; rng::Int=1)

    #@unpack T,L,R,ntrials = clicks
    ntrials = length(input_data)
    Random.seed!(rng)
    #rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    #binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)
    #@unpack nT, nL, nR = binned_clicks

    #should all of these be organized by trial? so they can be bundled and iterated over?
    output = pmap((input_data,rng) -> rand(θz,θy,input_data,
        f_str; rng=rng), input_data, shuffle(1:ntrials))

    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)

    return λ,a

end


"""
"""
function rand(θz::θz, θy::Vector{Vector{Float64}},
        input_data, f_str::String; rng::Int=1)

    @unpack λ0 = input_data

    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,λ0)-> map((a, λ0)-> f_py!(a, λ0, θy, f_str), a, λ0), θy, λ0)

    return λ, a

end


"""
"""
poisson_noise!(lambda,dt) = Int(rand(Poisson(lambda*dt)))
