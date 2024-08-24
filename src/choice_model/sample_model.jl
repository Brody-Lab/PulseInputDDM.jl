"""
    synthetic_data(; θ=θchoice(), ntrials=2000, rng=1)

Returns default parameters and ntrials of synthetic data (clicks and choices) organized into a choicedata type.
"""
function synthetic_data(; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, centered::Bool=false)

    clicks, choices = rand(θ, ntrials; rng=rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = map((clicks, binned_clicks)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        dt=dt, centered=centered), clicks, binned_clicks)

    return θ, choicedata.(inputs, choices)

end


"""
    rand(θ, ntrials)

Produces synthetic clicks and choices for n trials using model parameters θ.
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false)

    clicks = synthetic_clicks(ntrials, rng)
    binned_clicks = bin_clicks.(clicks,centered=centered,dt=dt)
    inputs = map((clicks, binned_clicks)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        dt=dt, centered=centered), clicks, binned_clicks)

    ntrials = length(inputs)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    #choices = rand.(Ref(θ), inputs, rng)
    choices = pmap((inputs, rng) -> rand(θ, inputs, rng), inputs, rng)

    return clicks, choices

end


"""
    rand(θ, inputs, rng)

Produces L/R choice for one trial, given model parameters and inputs.
"""
function rand(θ::θchoice, inputs::choiceinputs, rng::Int)

    Random.seed!(rng)
    @unpack θz, bias, lapse = θ

    a = rand(θz,inputs)
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

end