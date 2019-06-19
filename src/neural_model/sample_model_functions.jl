
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
    
    data["λ0_MC"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
    #data["spike_times"] = sample_spikes_single_session(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
    data["spike_counts_MC"] = sample_spikes_single_session(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
            
    return data
    
end

function sample_spikes_single_session(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)   
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC)
    #spike_times = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_spikes_single_trial(pz,py,λ0,nT,L,R,nL,nR;
    #    f_str=f_str, rng=rng, dtMC=dtMC), 
    #    data["λ0"], nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"]))) 
    
    spike_counts = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_spikes_single_trial(pz,py,λ0,nT,L,R,nL,nR;
        f_str=f_str, rng=rng, dtMC=dtMC), 
        data["λ0_MC"], nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))  
    
    return spike_counts
    
end

function sample_spikes_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int};
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)    
    a = sample_latent(nT,L,R,nL,nR,pz;dt=dtMC)
    #this assumes only one spike per bin, which should most often be true at 1e-4, but not guaranteed!
    #findall(x-> x > 1, pulse_input_DDM.poisson_noise!.(10 * ones(100 * 10 * Int(1. /1e-4)),1e-4))
    #Y = map((py,λ0)-> findall(x -> x != 0, 
    #        poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), dtMC)) .* dtMC, py, λ0)    
    
    Y = map((py,λ0)-> poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), dtMC), py, λ0)    
    
end

poisson_noise!(lambda,dt) = lambda = Int(rand(Poisson(lambda*dt)))

function sample_expected_rates_single_session(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", rng::Int=1)
    
    #removed lambda0_MC to lambda0
    Random.seed!(rng)   
    dt = data["dt"]
    output = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(pz,py,λ0,nT,L,R,nL,nR,dt;
        f_str=f_str, rng=rng), 
        data["λ0"], data["nT"], data["leftbups"], data["rightbups"], data["binned_leftbups"], 
        data["binned_rightbups"], shuffle(1:length(data["T"])))    
    
    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)  
    
    return λ,a
    
end

function sample_expected_rates_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, dt::Float64;
        f_str::String="softplus", rng::Int=1)
    
    Random.seed!(rng)  
    #changed this from 
    #a = decimate(sample_latent(nT,L,R,nL,nR,pz;dt=dtMC), Int(dt/dtMC))
    a = sample_latent(nT,L,R,nL,nR,pz;dt=dt)
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), py, λ0)  
    
    return λ, a
    
end

#################################### generate data with FP #########################

function sample_input_and_spikes_multiple_sessions(n::Int, pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        ntrials_per_sess::Vector{Int}; f_str::String="softplus", dtMC::Float64=1e-3)
    
    nsessions = length(ntrials_per_sess)
      
    data = map((py,ntrials,rng)-> sample_inputs_and_spikes_single_session(n, pz, py, ntrials; rng=rng, f_str=f_str, dtMC=dtMC), 
        py, ntrials_per_sess, 1:nsessions)   
    
    return data
    
end

function sample_inputs_and_spikes_single_session(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}}, ntrials::Int; 
        dtMC::Float64=1e-3, rng::Int=1, f_str::String="softplus")
    
    data = sample_clicks(ntrials; rng=rng)
    data["dtMC"] = dtMC
    data["N"] = length(py)
    
    data["λ0_MC"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
    #data["spike_times"] = sample_spikes_single_session(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
    data["spike_counts_MC"] = sample_spikes_single_session(n, data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
            
    return data
    
end

function sample_spikes_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-3, rng::Int=1)
    
    Random.seed!(rng)   
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC) 
    P,M,xc,dx, = initialize_latent_model(pz,n,dtMC)
    
    spike_counts = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_spikes_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,dx,xc;
        f_str=f_str, rng=rng, dtMC=dtMC), 
        data["λ0_MC"], nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))  
    
    return spike_counts
    
end

function sample_spikes_single_trial(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, 
        P, M, dx, xc;
        f_str::String="softplus", dtMC::Float64=1e-3, rng::Int=1)
    
    Random.seed!(rng)   
    a = sample_latent_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dtMC, n) 
                
    Y = map((py,λ0)-> poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), dtMC), py, λ0)    
    
end

function sample_latent_FP(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::VV,
        xc::Vector{WW},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        nL::Vector{Int}, nR::Vector{Int},dt::Float64,n::Int) where {UU,TT,VV,WW <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    a = Vector{TT}(undef,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T

        P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        P /= sum(P)
        
        a[t] = xc[findfirst(cumsum(P) .> rand())]
        #P = zeros(TT,n)
        #P[xc .== a[t]] .= one(TT)
        #should double check this it some point.
        P = TT.(xc .== a[t])
        
    end

    return a

end

function Pa_FP(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::VV,
        xc::Vector{WW},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        nL::Vector{Int}, nR::Vector{Int},
        dt::Float64,n::Int) where {UU,TT,VV,WW <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    Pa = Array{TT,2}(undef,n,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T

        P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        Pa[:,t] = P
        P /= sum(P)

    end

    return Pa

end

function sample_expected_rates_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", rng::Int=1)
    
    Random.seed!(rng)   
    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    output = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,dx,xc,dt;
        f_str=f_str, rng=rng), 
        data["λ0"], data["nT"], data["leftbups"], data["rightbups"], data["binned_leftbups"], 
        data["binned_rightbups"], shuffle(1:length(data["T"])))  
    
    λ = map(x-> x[1], output)
    a = map(x-> x[2], output)  

    return λ,a

    
end

function sample_expected_rates_single_trial(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}},
        λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, 
        P, M, dx, xc,
        dt::Float64; f_str::String="softplus", rng::Int=1)
    
    Random.seed!(rng)   
    a = sample_latent_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dt, n) 
                
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), py, λ0) 
    
    return λ, a
    
end

function Pa_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus")
    
    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    Pa = pmap((λ0,nT,L,R,nL,nR,rng) -> Pa_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,dx,xc,dt;
        f_str=f_str), 
        data["λ0"], data["nT"], data["leftbups"], data["rightbups"], data["binned_leftbups"], 
        data["binned_rightbups"], shuffle(1:length(data["T"])))  
    
    #λ = map(x-> x[1], output)
    #a = map(x-> x[2], output)  

    #return λ,a
    
    return Pa

    
end

function Pa_single_trial(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}},
        λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, 
        P, M, dx, xc,
        dt::Float64; f_str::String="softplus")
    
    Pa = Pa_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dt, n) 
                
    #λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), py, λ0) 
    
    #return λ, a
    
    return Pa
    
end
