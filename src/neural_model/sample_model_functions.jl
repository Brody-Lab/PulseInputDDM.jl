
#################################### Poisson neural observation model #########################

function sample_input_and_spikes_multiple_sessions(pz::Vector{Float64}, py::Vector{Vector{Float64}}, N, nsessions, 
    ntrials)
    
    
    #data["N"] = map(i->[1],1:ntrials);
    
    data = sample_single_session_inputs_and_spikes(pz, py, ntrials)
    
    return data
    
end

function sample_inputs_and_spikes_single_session(pz::Vector{Float64}, py::Vector{Float64}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int=1, f_str::String="softplus")
    
    Random.seed!(rng)
    data = sample_clicks(ntrials)         
    data = sample_spikes_single_session!(data, pz, py, dt; dtMC=dtMC, rng=rng, f_str=f_str)
            
    return data
    
end

function sample_spikes_single_session!(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}};
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)
    #data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_spikes_single_trial(pz,py,T,L,R;
    #        f_str=f_str, rng=rng, dtMC=dtMC), data["T"], data["leftbups"], data["rightbups"],
    #        shuffle(1:length(data["T"])));   
    
    data["spike_times"] = pmap((T,L,R,N,rng) -> sample_spikes_single_trial(pz,py,T,L,R;
        f_str=f_str, rng=rng, dtMC=dtMC), data["T"], data["leftbups"], data["rightbups"],
        shuffle(1:length(data["T"])))    
    
    return data
    
end

function sample_spikes_single_trial(pz, py, T::Float64, L::Vector{Float64}, R::Vector{Float64};
         f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1,
         λ0::Vector{Float64}=Vector{Float64}())
    
    Random.seed!(rng)    
    #removed 4/30, and moved towards unbinned spike times instead, then a binning procedure after this
    #A = decimate(sample_latent(T,L,R,pz;dt=dtMC), Int(dt/dtMC))   
    A = sample_latent(T,L,R,pz;dt=dtMC)
    Y = map(py-> poisson_noise.(fy22(py, A, λ0, f_str=f_str), dtMC), py)    
    #add a find here
    
end

poisson_noise(lambda,dt) = Int(rand(Poisson(lambda*dt)))

#=

function sampled_dataset!(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}},
        dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, num_reps::Int=1, rng::Int=1)

    construct_inputs!(data,num_reps)
    
    Random.seed!(rng)
    data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_model(pz,py,T,L,R,N,dt;
            f_str=f_str, rng=rng), data["T"], data["leftbups"], data["rightbups"],
            data["N"], shuffle(1:length(data["T"])));        
    
    return data
    
end

=#