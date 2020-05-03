"""
"""
function sample_clicks_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int;
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
    
    data = sample_clicks(ntrials;rng=rng)

    data["sessidx"] = Vector{Bool}(undef, ntrials)
    data["sessidx"][:] .= 0
    idx = 1
    data["sessidx"][idx] = 1
    for i = 1:ntrials/800
        idx = 400+ceil(Int,rand()*(800-400)) + idx
        data["sessidx"][idx] = 1
    end

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
        dtMC::Float64=1e-2, rng::Int = 1, use_bin_center::Bool=false)

    Random.seed!(rng)
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"]; dt=dtMC, use_bin_center=use_bin_center)

    σ2_i, B, B_λ, B_Δ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, η, α_prior, β_prior, B_0, γ_shape, γ_scale, γ_shape1, γ_scale1 = pz
    a_0 = compute_initial_value(data, η, α_prior, β_prior)
    a_0 = a_0 .+ B_0 # adding bias to initial point

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
    σ2_i, B, B_λ, B_Δ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, η, α_prior, β_prior, B_0, γ_shape, γ_scale, γ_shape1, γ_scale1 = pz

    choice = sign(a[RT]) > 0 

    if sign(a[RT]) > 0
        ndtime = Gamma(γ_shape1, γ_scale1)
        RT = round(RT*dtMC + rand(ndtime, 1)[1], digits = 4)
    else
        ndtime = Gamma(γ_shape, γ_scale)
        RT = round(RT*dtMC + rand(ndtime, 1)[1], digits = 4)
    end

    return choice, RT
    
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

    case = 1

    # L-R DBM
    if case == 1

        correct = data["correct"]

        α_actual = α_prior * β_prior
        β_actual = β_prior - α_actual

        prior = Beta(α_actual, β_actual)
        x = collect(0.001 : 0.001 : 1. - 0.001)
        prior_0 = pdf.(prior,x)
        prior_0 = prior_0/sum(prior_0)
        
        post = Array{Float64}(undef, size(prior_0))
        cprob = Array{TT}(undef, data["ntrials"]) 

        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
                prior_i = prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
                cprob[i] = Ep_x1_xt_1
            else
                prior_i = η*post + (1-η)*prior_0
                Ep_x1_xt_1 = sum(x.*prior_i)
           
                cprob[i] = Ep_x1_xt_1
            end

            if correct[i] == 1
                post = x.*prior_i
            else
                post = (1 .- x).* prior_i
            end
            post = post./sum(post)
        end

    # Alt-rep DBM    
    elseif case == 2

        correct = data["correct"]
        ra = abs.(diff(correct))
        ra = vcat(0, ra)

        α_actual = α_prior * β_prior
        β_actual = β_prior - α_actual

        prior = Beta(α_actual, β_actual)
        x = collect(0.001 : 0.001 : 1. - 0.001)
        prior_0 = pdf.(prior,x)
        prior_0 = prior_0/sum(prior_0)
        
        post = Array{Float64}(undef, size(prior_0))
        cprob = Array{TT}(undef, data["ntrials"]) 

        for i = 1:data["ntrials"]
            if data["sessidx"][i] == 1
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

    # exponential L-R    
    elseif case == 3


    # exponential Alt-Rep    
    elseif case == 4

        

    end

    return log.(cprob ./(1 .- cprob))

end
