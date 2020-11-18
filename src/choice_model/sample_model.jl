"""
    synthetic_data(; θ=θchoice(), ntrials=2000, rng=1)

Returns default parameters and ntrials of synthetic data (clicks and choices) organized into a choicedata type.
"""
function synthetic_data(; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, dt::Float64=1e-2, 
                            centered::Bool=false, initpt_mod::Bool=false)

    clicks, choices, sessbnd = rand(θ, ntrials; rng=rng, initpt_mod=initpt_mod)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = map((clicks, binned_clicks, sessbnd)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        sessbnd=sessbnd, dt=dt, centered=centered), clicks, binned_clicks, sessbnd)

    return θ, choicedata.(inputs, choices)

end


"""
    rand(θ, ntrials)

Produces synthetic clicks and choices for ntrials using model parameters θ.
"""
function rand(θ::θchoice, ntrials::Int; dt::Float64=1e-4, rng::Int = 1, centered::Bool=false, initpt_mod::Bool=false)

    clicks = synthetic_clicks(ntrials, rng)
    binned_clicks = bin_clicks.(clicks,centered=centered,dt=dt)
    sessbnd = [rand()<0.001 for i in 1:ntrials]
    sessbnd[1] = true
    inputs = map((clicks, binned_clicks, sessbnd)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        sessbnd=sessbnd, dt=dt, centered=centered), clicks, binned_clicks, sessbnd)

    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    #choices = rand.(Ref(θ), inputs, rng)
    choices = Array{Bool}(undef, ntrials)
    hits = Array{Bool}(undef, ntrials)
    correct = map(inputs -> Δclicks(inputs) > 0, inputs)
    # correct = map(data -> data.click_data.clicks.gamma > 0, data)

    @unpack θz, θlapse, θhist, bias = θ
    lim = 1

    for i = 1:ntrials
        Random.seed!(rng[i])

        if sessbnd[i] == true
            lim, i_0 = i, 0.
        else
            i_0 = compute_history(i, θhist, choices, hits, lim)                   
        end

        initpt_mod ? a = rand(θz,inputs[i], a_0 = i_0) : a = rand(θz,inputs[i])
        rlapse = get_rightlapse_prob(θlapse, i_0)
        rand() > θlapse.lapse_prob ? choices[i] = a[end] >= bias : choices[i] = rand()<rlapse
        hits[i] = choices[i] == correct[i]
    end

    return clicks, choices, sessbnd

end




"""
    synthetic_data(n; θ=θchoice(), ntrials=2000, rng=1)

Returns default parameters and ntrials of synthetic data (clicks and choices) organized into a choicedata type.
"""
function synthetic_data(dx::Float64; θ::θchoice=θchoice(), ntrials::Int=2000, rng::Int=1, 
                        dt::Float64=1e-2, centered::Bool=false, initpt_mod::Bool=false)

    clicks, choices, sessbnd = rand(θ, ntrials, dx; rng=rng, initpt_mod = initpt_mod, centered=centered, dt=dt)
    binned_clicks = bin_clicks.(clicks, centered=centered, dt=dt)
    inputs = map((clicks, binned_clicks, sessbnd)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        sessbnd = sessbnd, dt=dt, centered=centered), clicks, binned_clicks, sessbnd)

    return θ, choicedata.(inputs, choices)

end


"""
    rand(θ, ntrials, n)

Produces synthetic clicks and choices for ntrials using model parameters θ.
"""
function rand(θ::θchoice, ntrials::Int, dx::Float64; 
                dt::Float64=1e-2, rng::Int = 1, centered::Bool=false, initpt_mod::Bool=false)

    clicks = synthetic_clicks(ntrials, rng)
    binned_clicks = bin_clicks.(clicks,centered=centered,dt=dt)
    sessbnd = [rand()<0.001 for i in 1:ntrials]
    sessbnd[1] = true
    inputs = map((clicks, binned_clicks, sessbnd)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        sessbnd = sessbnd, dt=dt, centered=centered), clicks, binned_clicks, sessbnd)
    
    #θ = θ2(θ)

    @unpack θz, θhist, θlapse, bias = θ   
    @unpack σ2_i, B, λ, σ2_a = θz

    choices = Array{Bool}(undef, ntrials)
    hits = Array{Bool}(undef, ntrials)
    correct = map(inputs -> Δclicks(inputs) > 0, inputs)

    M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt)
    rng = sample(Random.seed!(rng), 1:ntrials, ntrials; replace=false)

    for i = 1:ntrials
        Random.seed!(rng[i])
    
        if sessbnd[i] == true
            lim, i_0 = i, 0.
        else
            i_0 = compute_history(i, θhist, choices, hits, lim)                   
        end

        initpt_mod ? a_0 = i_0 : a_0 = 0.
        P = P0(σ2_i, a_0, n, dx, xc, dt)
        P = P_single_trial!(θz,P,M,dx,xc,inputs[i],n,cross)   
        aend = xc[findfirst(cumsum(P) .> rand())]
        
        rlapse = get_rightlapse_prob(θlapse, i_0)
        rand() > θlapse.lapse_prob ? choice[i] = aend >= bias : choice[i] = rand()<rlapse

        hits[i] = choices[i] == correct[i]
    end

    return clicks, choices

end


function compute_history(i::Int, θhist::θtrialhist, choices, hits, lim::Int)

    @unpack h_ηc, h_ηe, h_βc, h_βe = θhist
    
    rel = max(lim, i-30):i-1
    rc = ((choices[rel] .== 1) .& (hits[rel] .== 1)).*h_ηc.*h_βc.^reverse(0:length(rel)-1)
    re = ((choices[rel] .== 1) .& (hits[rel] .== 0)).*h_ηe.*h_βe.^reverse(0:length(rel)-1)
    lc = ((choices[rel] .== 0) .& (hits[rel] .== 1)).*h_ηc.*h_βc.^reverse(0:length(rel)-1)
    le = ((choices[rel] .== 0) .& (hits[rel] .== 0)).*h_ηe.*h_βe.^reverse(0:length(rel)-1)
    i_0 = sum(rc - lc + re - le)

    return i_0
end


