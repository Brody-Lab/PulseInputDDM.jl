"""
"""
function sample_clicks_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int;
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)

    data = sample_clicks(ntrials;rng=rng)
    data["pokedR"] = sample_choices_all_trials(data, pz, pd; dtMC=dtMC, rng=rng, use_bin_center=use_bin_center)

    return data

end


"""
"""
function sample_choices_all_trials(data::Dict, pz::Vector{Float64}, pd::Vector{Float64};
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)

    Random.seed!(rng)
    binned = map((T,L,R)-> bin_clicks(T,L,R; dt=dtMC, use_bin_center=use_bin_center),
        data["T"], data["leftbups"], data["rightbups"])
    choices = pmap((L,R,binned,rng) -> sample_choice_single_trial(L,R,binned...,pz,pd;
            use_bin_center=use_bin_center, rng=rng), data["leftbups"], data["rightbups"], binned,
            shuffle(1:length(data["T"])))

end


"""
"""
function sample_choice_single_trial(L::Vector{Float64}, R::Vector{Float64},
        nT::Int, nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64}; use_bin_center::Bool=false, dtMC::Float64=1e-4, rng::Int=1)

    Random.seed!(rng)
    a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dtMC)

    bias,lapse = pd
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

end
