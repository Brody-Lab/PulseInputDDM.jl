"""
"""
function sample_clicks_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int;
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)

    data = sample_clicks(ntrials;rng=rng)

    if RTfit == true
        inp = sample_choices_all_trials(data, pz, pd; dtMC=dtMC, rng=rng, use_bin_center=use_bin_center)
        data["pokedR"] = map(i->inp[i][1],1:ntrials)
        data["T"] = map(i->inp[i][2],1:ntrials)
    else
        data["pokedR"] = sample_choices_all_trials(data, pz, pd; dtMC=dtMC, rng=rng, use_bin_center=use_bin_center)
    end

    return data

end


"""
"""
function sample_choices_all_trials(data::Dict, pz::Vector{Float64}, pd::Vector{Float64};
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)

    Random.seed!(rng)
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"]; dt=dtMC, use_bin_center=use_bin_center)

    if RTfit == true
        choices = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd;
                use_bin_center=use_bin_center, rng=rng), nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))
    else
        choices = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd;
                use_bin_center=use_bin_center, rng=rng), nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))
    end
end


"""
"""
function sample_choice_single_trial(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64}; use_bin_center::Bool=false, dtMC::Float64=1e-4, rng::Int=1)

    Random.seed!(rng)

    bias,lapse = pd

    if RTfit == true
        σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
        a, RT = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dtMC)
        choice = sign(a[RT]) > 0 
        return choice, RT*dtMC
    else
        a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dtMC)
        rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))
    end
end
