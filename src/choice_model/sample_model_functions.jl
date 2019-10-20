

function sample_inputs_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
    
    data = sample_clicks(ntrials)         
    choices = sample_choices_all_trials(data, pz, pd; dtMC=dtMC, rng=rng, use_bin_center=use_bin_center)
    
    data["pokedR"] = choices
            
    return data
    
end

function sample_choices_all_trials(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}; 
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
            
    Random.seed!(rng)
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC,use_bin_center)
    
    choices = pmap((nT,L,R,nL,nR,rng) -> sample_choice_single_trial(nT,L,R,nL,nR,pz,pd,use_bin_center,
            rng=rng),
        nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))
            
    return choices
    
end

function sample_choice_single_trial(nT::Int, L::Vector{Float64}, R::Vector{Float64},
        nL::Vector{Int}, nR::Vector{Int},
        pz::Vector{Float64},pd::Vector{Float64},use_bin_center::Bool;
        dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dtMC)
    
    bias,lapse = pd[1],pd[2]
            
    rand() > lapse ? choice = a[end] >= bias : choice = Bool(round(rand()))
    
end