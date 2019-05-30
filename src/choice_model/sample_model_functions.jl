
#################################### Choice observation model #################################

function sample_inputs_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int = 1)
    
    data = sample_clicks(ntrials)         
    data = sample_choices_all_trials!(data, pz, pd; dtMC=dtMC, rng=rng)
            
    return data
    
end

function sample_choices_all_trials!(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}; 
        dtMC::Float64=1e-4, rng::Int = 1)
            
    Random.seed!(rng)
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC)
    
    data["pokedR"] = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd,rng=rng),
        nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))
            
    return data
    
end

function sample_choice_single_trial(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64};dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    a = sample_latent(nT,L,R,nL,nR,pz;dt=dtMC)
    
    bias,lapse = pd[1],pd[2]
            
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))
    
end

#################################### New Choice model #################################

function sample_inputs_and_choices_new(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int = 1)
    
    data = sample_clicks(ntrials)         
    data = sample_choices_all_trials_new!(data, pz, pd; dtMC=dtMC, rng=rng)
            
    return data
    
end

function sample_choices_all_trials_new!(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}; 
        dtMC::Float64=1e-4, rng::Int = 1)
            
    Random.seed!(rng)
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC)
    
    data["pokedR"] = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial_new(nT,L,R,nL,nR,pz,pd,rng=rng),
        nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])));
            
    return data
    
end

function sample_choice_single_trial_new(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64};dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    a = sample_latent(nT,L,R,nL,nR,pz;dt=dtMC)
    
    Bernoulli_noise!(Bern_sig(pd,a[end]))
                
end

Bernoulli_noise!(p) = rand(Bernoulli(p))
