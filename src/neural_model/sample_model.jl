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
        dt::Float64=1e-2, rng::Int=0, dt_synthetic::Float64=1e-4)

    spikes,clicks,λ0 = rand(θ, nsess, ntrials, ncells; rng=rng)

    output = bin_clicks_spikes_λ0.(spikes, λ0, clicks;
        centered=centered, dt=dt, dt_synthetic=dt_synthetic, synthetic=true)

    spikes = map(x-> x[1], output)
    λ0 = map(x-> x[2], output)
    binned_clicks = map(x-> x[3], output)

    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered)

    neuraldata.(input_data,spikes,ncells)

    #spikes = bin_spikes.(spikes, dt; dt_synthetic=dt_synthetic, synthetic=true)
    #λ0 = bin_λ0.(λ0, dt; synthetic=true)
    #binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    #input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered)

    #neuraldata.(input_data,spikes,ncells)

end

#want to get this like detminsistic so can use the same, and want to make f for softplus etc.

"""
"""
synthetic_λ0(clicks, N::Int; dt::Float64=1e-4, rng::Int=1) = synthetic_λ0.(clicks, N; dt=dt, rng=rng)

"""
"""
function synthetic_λ0(clicks::clicks, N::Int; dt::Float64=1e-4, rng::Int=1)

    @unpack T = clicks

    #Random.seed!(rng)
    #data["λ0"] = [repeat([collect(range(10. *rand(),stop=10. * rand(),
    #                    length=Int(ceil(T./dt))))], outer=length(py)) for T in data["T"]]
    λ0 = repeat([zeros(Int(ceil(T/dt)))], outer=N)

end


"""
"""
function rand(θ::θneural, nsess, ntrials, ncells; centered::Bool=false, dt::Float64=1e-4, rng::Int=1)

    @unpack θy,θz = θ

    clicks = synthetic_clicks.(ntrials, collect((1:nsess) .+ rng))
    λ0 = synthetic_λ0.(clicks, ncells; dt=dt)

    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered)

    output = rand.(Ref(θz), θy, input_data; rng=rng)

    λ = map(x-> map(x-> x[1], x), output)
    Y = map(λ-> map(λ-> map(λ-> rand.(Poisson.(λ*dt)), λ), λ), λ)

    return Y, clicks, λ0

end


"""
"""
function rand(θz::θz, θy, input_data; rng::Int=1)

    ntrials = length(input_data)
    Random.seed!(rng)
    #rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((input_data,rng) -> rand(θz,θy,input_data; rng=rng), input_data, shuffle(1:ntrials))
    #pmap((input_data,rng) -> rand(θz,θy,input_data; rng=rng), input_data, rng)

end


"""
"""
function rand(θz::θz, θy, input_data::neuralinputs; rng::Int=1)

    @unpack λ0 = input_data

    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,λ0)-> θy(a, λ0), θy, λ0)

    return λ, a

end
