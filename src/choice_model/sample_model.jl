"""
    synthetic_data(;ntrials=2000, rng=1)
Returns default parameters and some simulated data
"""
function synthetic_data(; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, centered::Bool=false)

    clicks, choices = rand(θ, ntrials; rng=rng)
    binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)

    return θ, choicedata(binned_clicks, choices)

end


"""
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false)

    clicks = synthetic_clicks(ntrials; rng=rng)
    binned_clicks = bin_clicks(clicks,centered=centered,dt=dt)
    choices = rand(θ, binned_clicks; rng=rng)

    return clicks, choices

end


"""
"""
function rand(θ::θchoice, binned_clicks; rng::Int = 1)

    @unpack clicks, nT, nL, nR, dt, centered = binned_clicks
    @unpack L,R,ntrials = clicks

    Random.seed!(rng)
    #nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"]; dt=dtMC, use_bin_center=use_bin_center)
    #choices = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd;
    #        use_bin_center=use_bin_center, rng=rng), nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))

    #rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)
    pmap((nT,L,R,nL,nR,rng) -> rand(θ,nT,L,R,nL,nR;
            centered=centered, rng=rng, dt=dt), nT, L, R, nL, nR, shuffle(1:length(nT)))

end


"""
"""
function rand(θ::θchoice, nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int}; centered::Bool=false, dt::Float64=1e-4, rng::Int=1)

    Random.seed!(rng)
    @unpack θz, bias,lapse = θ

    a = rand(θz,nT,L,R,nL,nR;centered=centered, dt=dt)
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

end
