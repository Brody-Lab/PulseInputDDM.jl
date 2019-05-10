
#################################### Poisson neural observation model #########################

function sample_input_and_spikes_multiple_sessions(pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        ntrials_per_sess::Vector{Int}; f_str::String="softplus")
    
    nsessions = length(ntrials_per_sess)
      
    data = map((py,ntrials,rng)-> sample_inputs_and_spikes_single_session(pz, py, ntrials; rng=rng, f_str=f_str), 
        py, ntrials_per_sess, 1:nsessions)   
    
    return data
    
end

function sample_inputs_and_spikes_single_session(pz::Vector{Float64}, py::Vector{Vector{Float64}}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int=1, f_str::String="softplus")
    
    data = sample_clicks(ntrials; rng=rng)
    data["dtMC"] = dtMC
    data["N"] = length(py)
    
    data["λ0"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
    data["spike_times"] = sample_spikes_single_session(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
            
    return data
    
end

function sample_spikes_single_session(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)   
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC)
    spike_times = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_spikes_single_trial(pz,py,λ0,nT,L,R,nL,nR;
        f_str=f_str, rng=rng, dtMC=dtMC), 
        data["λ0"], nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))    
    
    return spike_times
    
end

function sample_spikes_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int};
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)    
    a = sample_latent(nT,L,R,nL,nR,pz;dt=dtMC)
    #this assumes only one spike per bin, which should most often be true at 1e-4, but not guaranteed!
    #findall(x-> x > 1, pulse_input_DDM.poisson_noise!.(10 * ones(100 * 10 * Int(1. /1e-4)),1e-4))
    Y = map((py,λ0)-> findall(x -> x != 0, 
            poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), dtMC)) .* dtMC, py, λ0)    
    
end

poisson_noise!(lambda,dt) = lambda = Int(rand(Poisson(lambda*dt)))

function sample_expected_rates_single_session(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)   
    dt = data["dt"]
    λ = map((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(pz,py,λ0,nT,L,R,nL,nR,dt;
        f_str=f_str, rng=rng, dtMC=dtMC), 
        data["λ0"], data["nT"], data["leftbups"], data["rightbups"], data["binned_leftbups"], 
        data["binned_rightbups"], shuffle(1:length(data["T"])))    
    
    return λ
    
end

function sample_expected_rates_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, dt::Float64;
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)    
    a = decimate(sample_latent(nT,L,R,nL,nR,pz;dt=dtMC), Int(dt/dtMC))
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), py, λ0)    
    
end
