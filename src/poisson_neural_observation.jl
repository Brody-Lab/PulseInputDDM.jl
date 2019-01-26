#module poisson_neural_observation

#using latent_DDM_common_functions, ForwardDiff, Optim, Pandas, Distributions
#using Distributed, SpecialFunctions, LinearAlgebra

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64}, fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, dt::Float64, n::Int; f_str::String="softplus", map_str::String="exp",
        beta::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0), 
        mu0::Vector{Vector{Float64}}=Vector{Vector{Float64}}(0)) where {TT}

    pz,py = breakup(gather(p_opt, p_const, fit_vec),f_str=f_str)
    map_pz!(pz,dt,map_str=map_str)       
    map_py!.(py,f_str=f_str)

    #11/5 changed this to deal with no priors, which is a default now
    #-(sum(LL_all_trials(pz, py, data, f_str=f_str, n=n, dt=dt)) - sum(gauss_prior.(py,mu0,beta)))
    LL = sum(LL_all_trials(pz, py, data, dt, n, f_str=f_str))
    
    length(beta) > 0 ? LL += sum(gauss_prior.(py,mu0,beta)) : nothing
    
    return -LL
              
end
    
function LL_all_trials(pz::Vector{TT},py::Union{Vector{Vector{TT}},Vector{Vector{Float64}}}, 
        data::Dict, dt::Float64, n::Int; f_str::String="softplus", comp_posterior::Bool=false) where {TT}
        
    P,M,xc,dx, = P_M_xc(pz,n,dt)
    
    lambday = fy.(py,xc',f_str=f_str)'
    #lambday = reshape(vcat(lambday...),n,:);           
                
    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, lambday[:,N], SC, dt, n, comp_posterior=comp_posterior),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["N"],data["spike_counts"])        
    
end

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
        comp_posterior::Bool=false) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #spike count data
    spike_counts = reshape(vcat(spike_counts...),:,length(spike_counts))
    
    c = Vector{TT}(undef,T)
    comp_posterior ? post = Array{Float64,2}(undef,n,T) : nothing
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = transition_Pa!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)        
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),dims=1)));
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's   
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            P .*= vec(exp.(sum(poiss_LL.(spike_counts[t+1,:],lambday',dt),dims=1)));           
            P,F = transition_Pa!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

function breakup(p; f_str::String="softplus")
                
    pz = p[1:dimz]

    if f_str == "sig"
        py = reshape(p[dimz+1:end],4,:)

    elseif f_str == "exp"
        py = reshape(p[dimz+1:end],2,:)

    elseif f_str == "softplus"
        py = reshape(p[dimz+1:end],3,:)

    end

    py = map(i->py[:,i],1:size(py,2))

    return pz, py
    
end

inv_breakup(pz::Vector{TT},py::Vector{Vector{TT}}) where {TT} = vcat(pz,vcat(py...))

function sampled_dataset!(data::Dict, p::Vector{Float64}, dt::Float64; 
        f_str::String="softplus", dtMC::Float64=1e-4, num_reps::Int=1, rng::Int=1)

    construct_inputs!(data,num_reps)
    
    srand(rng)
    data["spike_counts"] = pmap((T,L,R,N,rng) -> sample_model(p,T,L,R,N,dt;
            f_str=f_str, rng=rng), data["T"],data["leftbups"],data["rightbups"],
            data["N"], shuffle(1:length(data["T"])));        
    
    return data
    
end

function sample_model(p::Vector{Float64},T::Float64,L::Vector{Float64},R::Vector{Float64},
         N::Vector{Int}, dt::Float64; f_str::String="softplus", dtMC::Float64=1e-4, rng::Int=1,
         ts::Float64=0.,get_fr::Bool=false)
    
    srand(rng)
    
    pz,py = breakup(p,f_str=f_str)
    A = sample_latent(T,L,R,pz;dt=dtMC)
    A = cat(1,zeros(Int(ts/dtMC)),A)
    A = decimate(A,Int(dt/dtMC))      
    
    #this is if only you want the spike counts of one cell, which happens when I sample the model to get fake data
    #could probably find a better way to do this
    if length(N) > 1 || get_fr == false
        Y = map(py -> poisson_noise.(fy.([py],A,f_str=f_str),dt),py[N])
    else
        Y = poisson_noise.(fy.(py[N],A,f_str=f_str),dt)
    end
    
end

function map_py!(p::Vector{TT};f_str::String="softplus",map_str::String="exp") where {TT}
        
    if f_str == "exp"
        
        p[1] = exp(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = exp.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = 1e-5 + 99.99 * 0.5*(1+tanh.(p[1:2]))
            p[3:4] = -9.99 + 9.99*2 * 0.5*(1+tanh.(p[3:4]))
        end
        
    elseif f_str == "softplus"
          
        p[1] = exp(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function inv_map_py!(p::Vector{TT};f_str::String="softplus",map_str::String="exp") where {TT}
     
    if f_str == "exp"

        p[1] = log(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = log.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = atanh.(((p[1:2] - 1e-5)/(99.99*0.5)) - 1)
            p[3:4] = atanh.(((p[3:4] + 9.99)/(9.99*2*0.5)) - 1)
        end
        
    elseif f_str == "softplus"
        
        p[1] = log(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function fy(p::Vector{TT},a::Union{TT,Float64,Int};f_str::String="softplus",x::Float64=0.,mu::Float64=0.,std::Float64=0.) where {TT}
    
    if f_str == "sig"
        
        temp = p[3]*a + p[4]

        if exp(temp) < 1e-150
            y = p[1] + p[2]
        elseif exp(temp) >= 1e150
            y = p[1]
        else    
            y = p[1] + p[2]/(1. + exp(temp))
        end

        #protect from NaN gradient values
        #y[exp(temp) .<= 1e-150] = p[1] + p[2]
        #y[exp.(temp) .>= 1e150] = p[1]
        
    elseif f_str == "exp"
        
        y = p[1] + exp(p[2]*a)
        
    elseif f_str == "softplus"
            
        y = p[1] + log(1. + exp(p[2]*a + p[3]))
        
    elseif f_str == "expRBF"
            
        y = p[1] + exp(p[2]*a*pdf(Normal(mu,std),x))
        
    end
    
    return y
    
end

gauss_prior(p::Vector{TT}, mu::Vector{Float64}, beta::Vector{Float64}) where {TT} = -sum(beta .* (p - mu).^2)

"""
    poiss_LL(k,λ,dt)  

    returns poiss LL
"""
poiss_LL(k,λ,dt) = k*log(λ*dt) - λ*dt - lgamma(k+1)

poisson_noise(lambda,dt) = Int(rand(Poisson(lambda*dt)))

#end
