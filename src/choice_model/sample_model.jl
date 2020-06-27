"""
    synthetic_data(; θ=θchoice(), ntrials=2000, rng=1)

Returns default parameters and ntrials of synthetic data (clicks and choices) organized into a choicedata type.
"""
function synthetic_data(; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, centered::Bool=false)

    clicks, choices, sessbnd = rand(θ, ntrials; rng=rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = choiceinputs.(clicks, binned_clicks, dt, centered)

    return θ, choicedata.(inputs, choices, sessbnd)

end


"""
    rand(θ, ntrials)

Produces synthetic clicks and choices for n trials using model parameters θ.
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false)

    clicks = synthetic_clicks(ntrials, rng)
    binned_clicks = bin_clicks.(clicks,centered=centered,dt=dt)
    inputs = choiceinputs.(clicks, binned_clicks, dt, centered)

    ntrials = length(inputs)

    @unpack ibias, eta, beta, scaling = θ.θz
    sessbnd = [rand()<0.001 for i in 1:ntrials]
    i_0 = compute_initial_pt(ibias,eta,beta,scaling,inputs, sessbnd)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    #choices = rand.(Ref(θ), inputs, rng)
    choices = pmap((inputs, i_0, rng) -> rand(θ, inputs, i_0, rng), inputs, i_0, rng)

    return clicks, choices, sessbnd

end


"""
    rand(θ, inputs, rng)

Produces L/R choice for one trial, given model parameters and inputs.
# """
function rand(θ::θchoice, inputs::choiceinputs, i_0, rng::Int)

    Random.seed!(rng)
    @unpack θz, bias, lapse = θ
    
    a = rand(θz,inputs,i_0)
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand() < 1/(1+exp(-i_0))))

end
