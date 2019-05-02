
#################################### Poisson neural observation model #########################

function sample_input_and_spikes_multiple_sessions(pz::Vector{Float64}, py::Vector{Vector{Vector{Float64}}}, 
        ntrials_per_sess::Vector{Int})
    
      
    data = map((py,ntrials)-> sample_inputs_and_spikes_single_session(pz, py, ntrials), py, ntrials_per_sess)   
    
    return data
    
end

function sample_inputs_and_spikes_single_session(pz::Vector{Float64}, py::Vector{Vector{Float64}}, ntrials::Int; 
        dtMC::Float64=1e-4, rng::Int=1, f_str::String="softplus")
    
    Random.seed!(rng)
    data = sample_clicks(ntrials)
    data["dtMC"] = dtMC
    data["N"] = length(py)
    
    data["λ0"] = [repeat([zeros(Int(ceil(T./dtMC)))], outer=length(py)) for T in data["T"]]
    data = sample_spikes_single_session!(data, pz, py; dtMC=dtMC, rng=rng, f_str=f_str)
            
    return data
    
end

function sample_spikes_single_session!(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}}; 
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)
    
    data["spike_times"] = pmap((λ0,T,L,R,rng) -> sample_spikes_single_trial(pz,py,λ0,T,L,R;
        f_str=f_str, rng=rng, dtMC=dtMC), 
        data["λ0"], data["T"], data["leftbups"], data["rightbups"], shuffle(1:length(data["T"])))    
    
    return data
    
end

function sample_spikes_single_trial(pz::Vector{Float64}, py::Vector{Vector{Float64}}, λ0::Vector{Vector{Float64}}, 
        T::Float64, L::Vector{Float64}, R::Vector{Float64};
        f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1)
    
    Random.seed!(rng)    
    #removed 4/30, and moved towards unbinned spike times instead, then a binning procedure after this
    #A = decimate(sample_latent(T,L,R,pz;dt=dtMC), Int(dt/dtMC))   
    a = sample_latent(T,L,R,pz;dt=dtMC)
    #this assumes only one spike per bin, which should most often be true at 1e-4, but not guaranteed!
    #findall(x-> x > 1, pulse_input_DDM.poisson_noise!.(10 * ones(100 * 10 * Int(1. /1e-4)),1e-4))
    Y = map((py,λ0)-> findall(x -> x != 0, poisson_noise!.(f_py!(py, a, λ0, f_str=f_str), dtMC)) .* dtMC, py, λ0)    
    
end

poisson_noise!(lambda,dt) = lambda = Int(rand(Poisson(lambda*dt)))

function f_py!(p::Vector{T}, x::Vector{U}, c::Vector{Float64};
        f_str::String="softplus") where {T,U <: Any}

    if f_str == "sig"
    
        x = exp.((p[3] .* x .+ p[4]) + c)
        x[x .< 1e-150] .= p[1] + p[2]
        x[x .>= 1e150] .= p[1]
        x[(x .>= 1e-150) .& (x .< 1e150)] = p[1] .+ p[2] ./ (1. .+ x[(x .>= 1e-150) .& (x .< 1e150)])
        
    elseif f_str == "softplus"
        
        x = exp.((p[2] .* x .+ p[3]) + c)
        x[x .< 1e-150] .= eps() + p[1]
        x[x .>= 1e150] .= 1e150
        x[(x .>= 1e-150) .& (x .< 1e150)] = (eps() + p[1]) .+ log.(1. .+ x[(x .>= 1e-150) .& (x .< 1e150)])
        
    end

    return x
    
end

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