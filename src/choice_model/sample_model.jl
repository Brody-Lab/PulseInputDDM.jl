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

    # add a function here to compute initial point based on "corrects"
    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, η, α_prior, β_prior = pz
    a_0 = compute_initial_value(data, η, α_prior, β_prior)

    if RTfit == true
        choices = pmap((nT,L,R,nL,nR,a_0,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd,a_0;
                use_bin_center=use_bin_center, rng=rng), nT, data["leftbups"], data["rightbups"], nL, nR, a_0,shuffle(1:length(data["T"])))
    else
        choices = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd;
                use_bin_center=use_bin_center, rng=rng), nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))
    end
end


"""

this one is for RT

"""
function sample_choice_single_trial(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64}, a_0::TT; use_bin_center::Bool=false, dtMC::Float64=1e-4, rng::Int=1) where {TT <: Any}

    Random.seed!(rng)

    bias,lapse = pd

    a, RT = sample_latent(nT,L,R,nL,nR,pz,a_0,use_bin_center;dt=dtMC)
    choice = sign(a[RT]) > 0 
    return choice, RT*dtMC
    
end
"""
"""


function sample_choice_single_trial(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64}; use_bin_center::Bool=false, dtMC::Float64=1e-4, rng::Int=1)

    Random.seed!(rng)

    bias,lapse = pd
    a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dtMC)
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))

end


"""
"""
function compute_initial_value(data::Dict, η::TT, α_prior::TT, β_prior::TT) where {TT <: Any}

    correct = data["correct"]
    ra = abs.(diff(correct))
    ra = vcat(0, ra)

    prior = Beta(α_prior, β_prior)
    x = collect(0:0.01:1)
    prior_0 = pdf.(prior,x)
    prior_0 = prior_0/sum(prior_0)

    cprob = Array{TT}(undef, data["ntrials"]) 
    post = Array{Float64}(undef, size(prior_0))

    for i = 1:data["ntrials"]
        if i == 1
            prior_i = prior_0
            Ep_x1_xt_1 = sum(x.*prior_i)
            cprob[i] = Ep_x1_xt_1
        else
            prior_i = η*post + (1-η)*prior_0
            Ep_x1_xt_1 = sum(x.*prior_i)
            if correct[i-1] == 1
                cprob[i] = Ep_x1_xt_1
            else
                cprob[i] = 1-Ep_x1_xt_1
            end
        end

        if ra[i] == 1
            post = (1 .- x).* prior_i
        else
            post = x.*prior_i
        end
        post = post./sum(post)
    end

    return log.(cprob ./(1 .- cprob))

end