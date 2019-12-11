"""
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false)

    clicks = make_clicks(ntrials; rng=rng)
    binned_clicks = bin_clicks(clicks,centered=centered,dt=dt)
    model = choiceDDM(θ, binned_clicks)
    choices = rand(model; rng=rng)

    return clicks, choices

end


"""
"""
function rand(model::choiceDDM; rng::Int = 1)

    @unpack binned_clicks, θ = model
    @unpack clicks, nT, nL, nR, dt, centered = binned_clicks
    @unpack L,R,ntrials = clicks

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)
    pmap((nT,L,R,nL,nR,rng) -> rand(θ,nT,L,R,nL,nR;
            centered=centered, rng=rng, dt=dt), nT, L, R, nL, nR, rng)

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
