"""
"""
function synthetic_data(θ::θneural_choice,
        ntrials::Vector{Int}, ncells::Vector{Int}; centered::Bool=true,
        dt::Float64=1e-2, rng::Int=1, dt_synthetic::Float64=1e-4, 
        delay::Int=0, pad::Int=10, pos_ramp::Bool=false)

    nsess = length(ntrials)
    rng = sample(Random.seed!(rng), 1:nsess, nsess; replace=false)
    
    @unpack θz,θy,bias,lapse = θ

    output = rand.(Ref(θz), θy, bias, lapse, ntrials, ncells, rng; delay=delay, pad=0, pos_ramp=pos_ramp)

    spikes = getindex.(output, 1)
    λ0 = getindex.(output, 2)
    clicks = getindex.(output, 3)
    choices = getindex.(output, 4)

    output = bin_clicks_spikes_λ0.(spikes, clicks, λ0;
        centered=centered, dt=dt, dt_synthetic=dt_synthetic, synthetic=true)
    
    #λ0 = synthetic_λ0.(clicks, ncells; dt=dt, pos_ramp=pos_ramp, pad=0)

    spikes = getindex.(output, 1)
    binned_clicks = getindex.(output, 2)
    λ0 = getindex.(output, 3)

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
function rand(θz::θz, θy, bias, lapse, ntrials, ncells, rng; centered::Bool=false, dt::Float64=1e-4, pos_ramp::Bool=false, 
    delay::Int=0, pad::Int=10)

    clicks = synthetic_clicks.(ntrials, rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    λ0 = synthetic_λ0.(clicks, ncells; dt=dt, pos_ramp=pos_ramp, pad=pad)
    input_data = neuralinputs.(clicks, binned_clicks, λ0, dt, centered, delay, pad)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    output = pmap((input_data,rng) -> rand(θz,θy,bias,lapse,input_data; rng=rng), input_data, rng)
    
    spikes = getindex.(output, 3)
    choices = getindex.(output, 4)

    return spikes, λ0, clicks, choices

end


"""
"""
function rand(θz::θz, θy, bias, lapse, input_data::neuralinputs; rng::Int=1)

    @unpack λ0, dt = input_data

    Random.seed!(rng)
    a = rand(θz,input_data)
    λ = map((θy,λ0)-> θy(a, λ0), θy, λ0)
    spikes = map(λ-> rand.(Poisson.(λ*dt)), λ)   
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

    return λ, a, spikes, choice

end
