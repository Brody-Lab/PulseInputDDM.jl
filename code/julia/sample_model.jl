module sample_model

using Distributions, StatsBase, DSP
using global_functions, helpers

export sampled_dataset!
export sample_full_model

#eventually migrate these over to their respective modules, or global functions

function sampled_dataset!(data::Dict, pz::Vector{Float64},f_str::String; 
        dtMC::Float64=1e-4,dt::Float64=2e-2, num_reps::Int=1, 
        noise::String="Poisson",kwargs...)
    
    #still some work to do here, pretty sloppy. put downsample in function?
    kwargs = Dict(kwargs)  
    
    construct_inputs!(data,dt,num_reps)
            
    py = kwargs[:py]
    bias = kwargs[:bias]

    #this should only select neurons for which there was spikes on that trial, but double check
    output = pmap((T,leftbups,rightbups,N) -> sample_full_model(T,leftbups,rightbups,pz,py[N],bias,f_str;
            noise=noise,dt=dt),
        data["T"],data["leftbups"],data["rightbups"],data["N"]);

    data["spike_counts"] = map(x->x[1],output)
    data["pokedR"] = map(x->x[2],output);
    
    return data
    
end

#maybe can make data sampling fastest by putting nonlinearity and spikes in here 
function sample_full_model(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64},py::Vector{Vector{Float64}},
        bias::Float64,f_str::String;
        noise::String="Poisson",dtMC::Float64=1e-4,dt::Float64=2e-2,rng::Int=1)
    
    srand(rng)
    
    A = sample_latent(T,L,R,pz;dt=dtMC)
    A = decimate(A,Int(dt/dtMC))
            
    choice = A[end] >= bias;
    
    Y = map(py -> make_observation(lambda_y(A,py,f_str),dt=dt,noise=noise),py);

    return (Y,choice)
    
end

end
