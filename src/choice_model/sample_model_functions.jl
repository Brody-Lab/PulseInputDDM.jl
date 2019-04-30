
#################################### Choice observation model #################################

function sample_inputs_and_choices(pz::Vector{Float64}, pd::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int = 1)
    
    data = sample_clicks(ntrials)         
    data = sample_choices!(data, pz, pd; dtMC=dtMC, rng=rng)
            
    return data
    
end

function sample_choices!(data::Dict, pz::Vector{Float64}, pd::Vector{Float64}; 
        dtMC::Float64=1e-4, rng::Int = 1)
            
    Random.seed!(rng)
    data["pokedR"] = pmap((T,leftbups,rightbups,rng) -> sample_choice(T,leftbups,rightbups,pz,pd,rng=rng),
        data["T"],data["leftbups"],data["rightbups"], shuffle(1:length(data["T"])));
            
    return data
    
end

function sample_choice(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64},pd::Vector{Float64};dtMC::Float64=1e-4,rng::Int=1)
    
    Random.seed!(rng)
    
    A = sample_latent(T,L,R,pz;dt=dtMC)
    
    bias,lapse = pd[1],pd[2]
            
    rand() > lapse ? choice = A[end] >= bias : choice = Bool(round(rand()))
    
end