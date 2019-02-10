    
function LL_all_trials(pz::Vector{TT},py::Vector{Vector{TT}}, 
        data::Dict; dt::Float64=1e-2, n::Int=53, f_str::String="softplus", comp_posterior::Bool=false,
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT <: Any}
        
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    λ = hcat(fy.(py,[xc],f_str=f_str)...)
                    
    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, λ[:,N], SC, dt, n, λ0=λ0[N]),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["N"],data["spike_counts"])        
    
end

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        λ::Array{TT,2},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
        λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #construct T x N spike count array
    spike_counts = hcat(spike_counts...)
    
    c = Vector{TT}(undef,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks
    
    #construct T x N mean firing rate array
    λ0 = hcat(λ0...)

    @inbounds for t = 1:T
        
        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)        
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),dims=1)));
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],(log.(1. .+ exp.(lambday .+ lambda0')))',dt),dims=1)));
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],
                        transpose(softplus_3param([0.,1.,0.], λ .+ transpose(λ0[t,:]))), dt), dims=1)));
        c[t] = sum(P)
        P /= c[t] 

    end

    return sum(log.(c))

end

function posterior_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
        muf::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #spike count data
    spike_counts = reshape(vcat(spike_counts...),:,length(spike_counts))
    
    c = Vector{TT}(undef,T)
    post = Array{Float64,2}(undef,n,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)        
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),dims=1)));
        lambda0 = vcat(map(x->x[t],muf)...)
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],(log.(1. .+ exp.(lambday .+ lambda0')))',dt),dims=1)));
        c[t] = sum(P)
        P /= c[t] 
        post[:,t] = P

    end

    P = ones(Float64,n); #initialze backward pass with all 1's   
    post[:,T] .*= P;

    @inbounds for t = T-1:-1:1

        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t+1,:],lambday',dt),dims=1)));           
        P,F = latent_one_step!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
        P /= c[t+1] 
        post[:,t] .*= P

    end

    return post

end

"""
    poiss_LL(k,λ,dt)  

    returns poiss LL
"""
poiss_LL(k,λ,dt) = k*log(λ*dt) - λ*dt - lgamma(k+1)

function fy(p::Vector{T},a::Vector{U}; f_str::String="softplus") where {T,U <: Any}
    
    if (f_str == "sig") || (f_str == "sig2")
        
        y = sigmoid_4param(p,a)
        
    elseif f_str == "exp"
        
        y = p[1] + exp(p[2]*a)
        
    elseif f_str == "softplus"
        
        y = softplus_3param(p,a)
                    
    end
        
end

function sigmoid_4param(p::Vector{T},x::Vector{U}) where {T,U <: Any}
    
    y = exp.(p[3] .* x .+ p[4]) 
    y[y .< 1e-150] .= p[1] + p[2]
    y[y .>= 1e150] .= p[1]
    y[(y .>= 1e-150) .& (y .< 1e150)] = p[1] .+ p[2] ./ (1. .+ y[(y .>= 1e-150) .& (y .< 1e150)])
    
    return y
    
end

softplus_3param(p::Vector{T}, x::Array{U}) where {T,U <: Any} = p[1] .+ log.(1. .+ exp.(p[2] .* x .+ p[3])) 

########################## Model with RBF #################################################################

function LL_all_trials(pz::Vector{TT}, py::Vector{Vector{TT}}, pRBF::Vector{Vector{TT}}, 
        data::Dict; dt::Float64=1e-2, n::Int=53, 
        f_str::String="softplus", comp_posterior::Bool=false) where {TT <: Any}
        
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    λ = hcat(fy.(py,[xc],f_str=f_str)...)
    c = map(x->dt:dt:maximum(data["nT"][x])*dt,data["trial"])
    rbf = map(x->UniformRBFE(x,num_RBFs),c);
    λ0 = map((x,y,z)->x(y)*z, rbf, c, pRBF)
                    
    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, λ[:,N], SC, dt, n, λ0=λ0[N]),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["N"],data["spike_counts"])        
    
end
