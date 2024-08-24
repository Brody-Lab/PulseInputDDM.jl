#################################### generate data with FP #########################

function sample_input_and_spikes_multiple_sessions(n::Int, pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        ntrials_per_sess::Vector{Int}; f_str::String="softplus", dtMC::Float64=1e-4, use_bin_center::Bool=false)
    
    nsessions = length(ntrials_per_sess)
      
    data = map((py,ntrials,rng)-> sample_inputs_and_spikes_single_session(n, pz, py, ntrials; rng=rng, f_str=f_str, 
            dtMC=dtMC, use_bin_center=use_bin_center), 
        py, ntrials_per_sess, 1:nsessions)   
    
    return data
    
end

function sample_inputs_and_spikes_single_session(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}}, ntrials::Int; 
        dtMC::Float64=1e-2, rng::Int=1, f_str::String="softplus", use_bin_center::Bool=false)
    
    data = sample_clicks(ntrials; rng=rng)
    data["dtMC"] = dtMC
    data["N"] = length(py)
    
    data["λ0_MC"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
    #data["spike_times"] = sample_spikes_single_session(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
    data["spike_counts_MC"] = sample_spikes_single_session(n, data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str,
        use_bin_center=use_bin_center)
            
    return data
    
end

function sample_spikes_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-2, rng::Int=1, use_bin_center::Bool=false)
    
    Random.seed!(rng)   
    nT,nL,nR = bin_clicks(data["T"],data["leftbups"],data["rightbups"],dtMC,use_bin_center) 
    P,M,xc,dx, = initialize_latent_model(pz,n,dtMC)
    
    spike_counts = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_spikes_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,dx,xc;
        f_str=f_str, rng=rng, dtMC=dtMC, use_bin_center=use_bin_center), 
        data["λ0_MC"], nT, data["leftbups"], data["rightbups"], nL, nR, shuffle(1:length(data["T"])))  
    
    return spike_counts
    
end

function sample_spikes_single_trial(n::Int, pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        nT::Int, L::Vector{Float64}, R::Vector{Float64}, nL::Vector{Int}, nR::Vector{Int}, 
        P, M, dx, xc;
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1, use_bin_center::Bool=false)
    
    Random.seed!(rng)   
    a = sample_latent_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dtMC, n, use_bin_center) 
                
    Y = map((py,λ0)-> poisson_noise!.(map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), dtMC), py, λ0)    
    
end


#################################### Expected rates, FP #########################

function sample_expected_rates_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", rng::Int=1)
    
    Random.seed!(rng)   
    dt = data["dt"]
    use_bin_center = data["use_bin_center"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    output = pmap((λ0,nT,L,R,nL,nR,rng) -> sample_expected_rates_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,
            dx,xc,dt,use_bin_center;
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
        dt::Float64, use_bin_center::Bool; f_str::String="softplus", rng::Int=1)
    
    Random.seed!(rng)   
    a = sample_latent_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dt, n, use_bin_center) 
                
    λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), py, λ0) 
    
    return λ, a
    
end

#################################### Pa, FP #########################

function Pa_single_session(n::Int, data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus")
    
    dt = data["dt"]
    use_bin_center = data["use_bin_center"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    Pa = pmap((λ0,nT,L,R,nL,nR,rng) -> Pa_single_trial(n,pz,py,λ0,nT,L,R,nL,nR,P,M,dx,xc,dt,use_bin_center;
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
        dt::Float64,use_bin_center::Bool; f_str::String="softplus")
    
    Pa = Pa_FP(pz, P, M, dx, xc, L, R, nT, nL, nR, dt, n, use_bin_center) 
                
    #λ = map((py,λ0)-> map((a, λ0)-> f_py!(a, λ0, py, f_str=f_str), a, λ0), py, λ0) 
    
    #return λ, a
    
    return Pa
    
end

function Pa_FP(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::VV,
        xc::Vector{WW},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        nL::Vector{Int}, nR::Vector{Int},
        dt::Float64,n::Int,use_bin_center::Bool) where {UU,TT,VV,WW <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    Pa = Array{TT,2}(undef,n,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T

        if use_bin_center && t == 1 
            P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt/2)
        else
            P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        end
        
        Pa[:,t] = P
        P /= sum(P)

    end

    return Pa

end
