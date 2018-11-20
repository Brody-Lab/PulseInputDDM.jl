module sample_model

using Distributions, StatsBase
using global_functions, helpers

export sample_latent, make_observation, sampled_dataset!
export downsample_spiketrain, sample_full_model

function sampled_dataset!(data::Dict, pz::Vector{Float64}; 
        dtMC::Float64=1e-4,dt::Float64=2e-2, num_reps::Int=1, rng::Int=1, 
        noise::String="Poisson",kwargs...)
    
    #still some work to do here, pretty sloppy. put downsample in function?
    kwargs = Dict(kwargs)  
    
    construct_inputs!(data,dt,num_reps)
    
    #this will not create the same data each time because of the pmap
    srand(rng)

    #sample latent paths
    #A_sampled = pmap((T,leftbups,rightbups) -> sample_latent(T,leftbups,rightbups,pz,path=true),
    #    data["T"],data["leftbups"],data["rightbups"]);
    
    if haskey(kwargs, :py) && haskey(kwargs, :bias)
        
        py = kwargs[:py]
        bias = kwargs[:bias]
    
        #this should only select neurons for which there was spikes on that trial, but double check
        output = pmap((T,leftbups,rightbups,N) -> sample_full_model(T,leftbups,rightbups,pz,py[N],bias;
                noise=noise,dt=dt),
            data["T"],data["leftbups"],data["rightbups"],data["N"]);
        
        data["spike_counts"] = map(x->x[1],output)
        data["pokedR"] = map(x->x[2],output);
        
    elseif haskey(kwargs, :py)
        
        py = kwargs[:py]
    
        #this should only select neurons for which there was spikes on that trial, but double check
        output = pmap((T,leftbups,rightbups,N) -> sample_full_model(T,leftbups,rightbups,pz,py[N],0.;
                noise=noise,dt=dt),
            data["T"],data["leftbups"],data["rightbups"],data["N"]);
        
        data["spike_counts"] = map(x->x[1],output)
        
    elseif haskey(kwargs, :bias)
        
        bias = kwargs[:bias]
        
        data["pokedR"] = pmap((T,leftbups,rightbups) -> sample_choice_model(T,leftbups,rightbups,pz,bias),
            data["T"],data["leftbups"],data["rightbups"]);
        
        #nbins = Int(dt/dtMC)
    
        #compute firing rates
        #lambda_sampled = pmap(a -> map(py -> my_sigmoid(a,py),py),A_sampled);
        #can downsample lambda too
        #data["lambda"] = lambda_sampled;
        #generate spikes
        #spikes_sampled = pmap(j -> map(i -> make_spikes(lambda_sampled[j][i]),data["N"][j]),1:length(lambda_sampled));
        #data["spike_counts"] = pmap(k -> map(i -> downsample_spiketrain(k[i],nbins),1:length(k)),spikes)
        
        #bin and downsample spikes
        #for i = 1:length(spikes_sampled)
        #    spikes_sampled[i] = cat(1,spikes_sampled[i],
        #            zeros(nbins - rem(size(spikes_sampled[i],1),nbins),
        #            size(spikes_sampled[i],2)))
        #    data["spike_counts"][i] = squeeze(sum(reshape(spikes_sampled[i],nbins,
        #                    Int(size(spikes_sampled[i],1)/nbins),size(spikes_sampled[i],2)),1),1)     
#
        #end
        
    end
    
    #make new choices
    #haskey(data, :bias) ? (bias = kwargs[:bias]; data["pokedR"] = pmap(x->x[end]>=bias,A_sampled)) : nothing
    #for i = 1:length(data["pokedR"])
    #    data["pokedR"][i] = A_sampled[i][end] >= bias
    #end
    
    return data
    
end

function construct_inputs!(data::Dict,dt::Float64,num_reps::Int)
    
    binnedT = ceil.(Int,data["T"]/dt);

    data["nT"] = binnedT
    data["binned_leftbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["leftbups"])
    data["binned_rightbups"] = map((x,y)->vec(qfind(0.:dt:x*dt,y)),binnedT,data["rightbups"])
    
    #use repmat to make as any copys as needed
    data["nT"] = repmat(data["nT"],num_reps)
    data["binned_leftbups"] = repmat(data["binned_leftbups"],num_reps)
    data["binned_rightbups"] = repmat(data["binned_rightbups"],num_reps)
    data["N"] = repmat(data["N"],num_reps)
    data["T"] = repmat(data["T"],num_reps)
    data["leftbups"] = repmat(data["leftbups"],num_reps)
    data["rightbups"] = repmat(data["rightbups"],num_reps)
    data["trial0"] = data["trial0"] * num_reps;
    
    return data
    
end

#maybe can make data sampling fastest by putting nonlinearity and spikes in here 
function sample_choice_model(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64},bias::Float64;
        noise::String="Poisson",dtMC::Float64=1e-4,dt::Float64=2e-2)
    
    A = sample_latent(T,L,R,pz;dt=dtMC)
            
    choice = A[end] >= bias;

    return choice
    
end


#maybe can make data sampling fastest by putting nonlinearity and spikes in here 
function sample_full_model(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64},py::Vector{Vector{Float64}},
        bias::Float64;
        noise::String="Poisson",dtMC::Float64=1e-4,dt::Float64=2e-2)
    
    A = sample_latent(T,L,R,pz;dt=dtMC)
            
    choice = A[end] >= bias;
    Y = map(py -> make_observation(my_sigmoid(A,py),dtMC=dtMC,dt=dt,noise=noise),py);

    return (Y,choice)
    
end

function sample_latent(T::Float64,L::Vector{Float64},R::Vector{Float64},
        pz::Vector{Float64};dt::Float64=1e-4)
    
    vari, inatt, B, lambda, vara, vars, phi, tau_phi = pz;
    
    nT = Int(ceil.(T/dt)); # number of timesteps
                    
    if rand() < inatt

        a = B * sign(randn())
        A = repmat(a,nT)

    else

        La, Ra = make_adapted_clicks(phi,tau_phi,L,R)
        t = 0.:dt:nT*dt-dt; 
        hereL = vec(qfind(t,L))
        hereR = vec(qfind(t,R))
         
        A = Vector{Float64}(nT)
        a = sqrt(vari)*randn()

        for t = 1:nT

            #inputs
            any(t .== hereL) ? sL = sum(La[t .== hereL]) : sL = zero(phi)
            any(t .== hereR) ? sR = sum(Ra[t .== hereR]) : sR = zero(phi)
            var = vars * (sL + sR)  
            mu = -sL + sR
            (sL + sR) > 0. ? a += mu + sqrt(var) * randn() : nothing

            #drift and diffuse
            a += (dt*lambda) * a + sqrt(vara * dt) * randn();

            abs(a) > B ? (a = B * sign(a); A[t:nT] = a; break) : A[t] = a

        end               
    end
    
    return A
    
end

function make_observation(lambda;dtMC::Float64=1e-4,dt::Float64=2e-2,noise::String="Poisson")
    
    lambda = downsample_spiketrain(lambda;dtMC=dtMC,dt=dt);
    
    #spikes = Array{Int}(size(lambda))
    
    #for i = 1:length(lambda)
    #    spikes[i] = Int(rand(Poisson(lambda[i]*dt)))
    #end
    
    if noise == "Poisson"
        y = map(lambda->Int(rand(Poisson(lambda*dt))),lambda)
    elseif noise == "Gaussian"
        y = map(lambda->rand(Normal(lambda,1e-6)),lambda)
    end
    
    return y
    
end

function downsample_spiketrain(x;dtMC::Float64=1e-4,dt::Float64=2e-2)
    
    nbins = Int(dt/dtMC)    
    x = vcat(x, mean(x[end-rem(length(x),nbins)+1:end]) * ones(eltype(x),mod(nbins - rem(length(x),nbins),nbins)))
    x = squeeze(mean(reshape(x,nbins,:),1),1)  
    
    return x
    
end

end
