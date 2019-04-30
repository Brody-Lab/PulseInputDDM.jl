
#################################### Poisson neural observation model #########################

function sampled_dataset!(data::Dict, pz::Vector{Float64}, py::Vector{Vector{Float64}},
        dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, num_reps::Int=1, rng::Int=1)

    construct_inputs!(data,num_reps)
    
    Random.seed!(rng)
    data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_model(pz,py,T,L,R,N,dt;
            f_str=f_str, rng=rng), data["T"], data["leftbups"], data["rightbups"],
            data["N"], shuffle(1:length(data["T"])));        
    
    return data
    
end

function sample_model(pz, py, T::Float64, L::Vector{Float64}, R::Vector{Float64},
         N::Vector{Int}, dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1,
         λ0::Vector{Float64}=Vector{Float64}())
    
    Random.seed!(rng)
    
    A = decimate(sample_latent(T,L,R,pz;dt=dtMC), Int(dt/dtMC))  
    
    Y = map(py-> poisson_noise.(fy22(py, A, λ0, f_str=f_str), dt), py[N])       
    
end

poisson_noise(lambda,dt) = Int(rand(Poisson(lambda*dt)))