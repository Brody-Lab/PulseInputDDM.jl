function simulate_expected_firing_rate(θ::Union{θneural, θneural_choice}, data, rng)

    @unpack θz,θy = θ
    μ_λ = rand.(Ref(θz), θy, data, Ref(rng))
        
    return μ_λ

end


"""
    simulate_expected_firing_rate(model)

Given a `model` generate samples of the firing rate `λ` of each neuron.

Arguments:

- `model`: an instance of a `neuralDDM`.

Optional arguments:

- `num_samples`: How many independent samples of the latent to simulate, to average over.

Returns:

- ` λ`: an `array` is length `num_samples`. Each entry an `array` of length number of trials. Each entry of that is an `array` of length number of neurons. Each entry of that is the firing rate of that neuron on that trial for some length of time.
- `μ_λ`: the mean firing rate for each neuron (averaging across the noise of the latent process for `num_samples` trials). 
- `μ_c_λ`: the average across trials and across group with similar evidence values (grouped into `nconds` number of groups).

"""
function simulate_expected_firing_rate(model; num_samples::Int=100, nconds::Int=2, rng1::Int=1)

    @unpack θ,data = model
    @unpack θz,θy = θ
    
    rng = sample(Random.seed!(rng1), 1:num_samples, num_samples; replace=false)
    λ = map(rng-> rand.(Ref(θz), θy, data, Ref(rng)), rng)
    μ_λ = mean(λ)
    
    μ_c_λ = cond_mean.(μ_λ, data; nconds=nconds)
    
    return μ_λ, μ_c_λ, λ

end


function simulate_expected_spikes(model; num_samples::Int=100, nconds::Int=2, rng1::Int=1)

    @unpack θ,data = model
    @unpack θz,θy = θ
    
    rng = sample(Random.seed!(rng1), 1:num_samples, num_samples; replace=false)
    spikes = map(rng-> rand_spikes.(Ref(θz), θy, data, Ref(rng)), rng)
    μ_spikes = mean(spikes)
    
    μ_c_spikes = cond_mean.(μ_spikes, data; nconds=nconds)
    
    return μ_spikes, μ_c_spikes, spikes

end


"""
    Sample all trials over one session
"""
function rand_a(θz::θz, θy, data, rng)
    
    ntrials = length(data)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((data,rng) -> rand(θz,θy,data.input_data; rng=rng)[2], data, rng)

end


"""
    Sample all trials over one session
"""
function rand_spikes(θz::θz, θy, data, rng)
    
    ntrials = length(data)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((data,rng) -> rand(θz,θy,data.input_data; rng=rng)[3], data, rng)

end


"""
    Sample all trials over one session
"""
function rand(θz::θz, θy, data, rng)
    
    ntrials = length(data)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    pmap((data,rng) -> rand(θz,θy,data.input_data; rng=rng)[1], data, rng)

end


"""
"""
function cond_mean(μ_λ, data; nconds=2)
    
    ncells = data[1].ncells
        
    pad = data[1].input_data.pad
    nT = map(x-> x.input_data.binned_clicks.nT, data)
    ΔLRT = map((data,nT) -> getindex(diffLR(data), pad+nT), data, nT)
    conds = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), ΔLRT)), ΔLRT)

    map(n-> map(c-> [mean([μ_λ[conds .== c][k][n][t]
        for k in findall(nT[conds .== c] .+ (2*pad) .>= t)])
        for t in 1:(maximum(nT[conds .== c] .+ (2*pad)))],
                1:nconds), 1:ncells)

end


"""
"""
function synthetic_data(θ::θneural,
        ntrials::Vector{Int}, ncells::Vector{Int}; centered::Bool=true,
        dt::Float64=1e-2, rng::Int=1, dt_synthetic::Float64=1e-4, 
        delay::Int=0, pad::Int=10, pos_ramp::Bool=false)

    nsess = length(ntrials)
    rng = sample(Random.seed!(rng), 1:nsess, nsess; replace=false)
    
    @unpack θz,θy = θ

    output = rand.(Ref(θz), θy, ntrials, ncells, rng; delay=delay, pad=0, pos_ramp=pos_ramp)

    spikes = getindex.(output, 1)
    λ0 = getindex.(output, 2)
    clicks = getindex.(output, 3)
    choices = getindex.(output, 4)

    output = bin_clicks_spikes_λ0.(spikes, clicks, λ0;
        centered=centered, dt=dt, dt_synthetic=dt_synthetic, synthetic=true)
    
    λ0 = synthetic_λ0.(clicks, ncells; dt=dt, pos_ramp=pos_ramp, pad=0)

    spikes = getindex.(output, 1)
    binned_clicks = getindex.(output, 2)

    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered, delay, 0)
    
    padded = map(spikes-> map(spikes-> map(SCn-> vcat(rand.(Poisson.((sum(SCn[1:10])/(10*dt))*ones(pad)*dt)), 
                    SCn, rand.(Poisson.((sum(SCn[end-9:end])/(10*dt))*ones(pad)*dt))), spikes), spikes), spikes)
    
    μ_rnt = map(padded-> filtered_rate.(padded, dt), padded)
    
    nT = map(x-> map(x-> x.nT, x), binned_clicks)
    
    μ_t = map((μ_rnt, ncells, nT)-> map(n-> [max(0., mean([μ_rnt[i][n][t]
        for i in findall(nT .>= t)]))
        for t in 1:(maximum(nT))], 1:ncells), μ_rnt, ncells, nT)

    neuraldata.(input_data, spikes, ncells, choices), μ_rnt, μ_t

end


"""
"""
synthetic_λ0(clicks, N::Int; dt::Float64=1e-4, rng::Int=1, pos_ramp::Bool=false, pad::Int=10) = 
    synthetic_λ0.(clicks, N; dt=dt, rng=rng, pos_ramp=pos_ramp, pad=pad)


"""
"""
function synthetic_λ0(clicks::clicks, N::Int; dt::Float64=1e-4, rng::Int=1, pos_ramp::Bool=false, pad::Int=10)

    @unpack T = clicks

    Random.seed!(rng)
    if pos_ramp
        λ0 = repeat([collect(range(10. + 5*rand(), stop=20. + 5*rand(), length=Int(ceil(T/dt))))], outer=N)
    else
        λ0 = repeat([zeros(Int(ceil(T/dt) + 2*pad))], outer=N)
    end

end


"""
"""
function rand(θz::θz, θy, ntrials, ncells, rng; centered::Bool=false, dt::Float64=1e-4, pos_ramp::Bool=false, 
    delay::Int=0, pad::Int=10)

    clicks = synthetic_clicks.(ntrials, rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    λ0 = synthetic_λ0.(clicks, ncells; dt=dt, pos_ramp=pos_ramp, pad=pad)
    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered, delay, pad)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    output = pmap((input_data,rng) -> rand(θz,θy,input_data; rng=rng), input_data, rng)
    
    spikes = getindex.(output, 3)
    choices = getindex.(output, 4)

    return spikes, λ0, clicks, choices

end


"""
"""
function rand(θz::θz, θy, input_data::neuralinputs; rng::Int=1)

    @unpack λ0, dt = input_data

    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,λ0)-> θy(a, λ0), θy, λ0)
    spikes = map(λ-> rand.(Poisson.(λ*dt)), λ)   
    choice = a[end] .> 0.

    return λ, a, spikes, choice

end
