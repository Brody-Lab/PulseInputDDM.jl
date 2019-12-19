"""
    synthetic_data(;ntrials=2000, rng=1)
Returns default parameters and some simulated data
"""
function synthetic_data(; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, centered::Bool=false)

    clicks, choices = rand(θ, ntrials; rng=rng)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = click_data.(clicks, binned_clicks, dt, centered)

    return θ, choicedata.(inputs, choices)

end


"""
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false)

    clicks = synthetic_clicks(ntrials; rng=rng)
    binned_clicks = bin_clicks.(clicks,centered=centered,dt=dt)
    inputs = click_data.(clicks, binned_clicks, dt, centered)
    choices = rand(θ, inputs; rng=rng)

    return clicks, choices

end


"""
"""
function rand(θ::θchoice, click_data; rng::Int = 1)

    ntrials = length(click_data)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)
    pmap((click_data, rng) -> rand(θ,click_data; rng=rng), click_data, rng)

end


"""
"""
function rand(θ::θchoice, click_data::click_data; rng::Int=1)

    Random.seed!(rng)
    @unpack θz, bias,lapse = θ

    a = rand(θz,click_data)
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

end
