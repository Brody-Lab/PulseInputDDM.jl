
function LL_all_trials(pz::Vector{TT}, py::Vector{Vector{TT}}, data::Dict, 
        n::Int; f_str::String="softplus") where {TT <: Any}
     
    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    #λ = hcat(fy.(py,[xc],f_str=f_str)...)
                    
    #output = pmap((L,R,T,nL,nR,N,SC,λ0) -> LL_single_trial(pz, P, M, dx, xc,
    #    L, R, T, nL, nR, λ[:,N], SC, dt, n, λ0=λ0),
    #    data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
    #    data["binned_rightbups"], data["N"],data["spike_counts"],λ0)   
                            
    output = pmap((L,R,T,nL,nR,SC,λ0) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, py, SC, dt, n, λ0, f_str=f_str),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["spike_counts"], data["λ0"])   
    
end

#function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT,
#        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
#        hereL::Vector{Int}, hereR::Vector{Int},
#        λ::Array{TT,2},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
#        λ0::Vector{Vector{UU}}=Vector{Vector{UU}}()) where {UU,TT <: Any}

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT,
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        py::Vector{Vector{TT}},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int,
        λ0::Vector{Vector{UU}};
        f_str::String="softplus") where {UU,TT <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #construct T x N spike count array
    spike_counts = hcat(spike_counts...)

    c = Vector{TT}(undef,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks
    
    #λ = hcat(fy.(py,[xc],f_str=f_str)...)

    #construct T x N mean firing rate array
    λ0 = hcat(λ0...)

    @inbounds for t = 1:T

        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],lambday',dt),dims=1)));
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],(log.(1. .+ exp.(lambday .+ lambda0')))',dt),dims=1)));
        #P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:],
        #                transpose(softplus_0param(λ .+ transpose(λ0[t,:]))), dt), dims=1)))
        
        #y = hcat(map((py,c)-> fy2(py,xc,c), py, λ0[t,:])...)
        #y = hcat(map((py,c)-> softplus_3param2(py,xc,c), py, λ0[t,:])...)
         y = hcat(map((py,c)-> fy2(py,xc,c, f_str=f_str), py, λ0[t,:])...)
        
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:], transpose(y), dt), dims=1)))
        
        c[t] = sum(P)
        P /= c[t]

    end

    return sum(log.(c))

end

function PY_all_trials(pz::Vector{TT},py::Vector{Vector{TT}}, 
        data::Dict; dt::Float64=1e-2, n::Int=53, f_str::String="softplus", comp_posterior::Bool=false,
        λ0::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}()) where {TT <: Any}
        
    P,M,xc,dx, = initialize_latent_model(pz,n,dt) 
                        
    output = pmap((L,R,T,nL,nR,N,SC,λ0) -> PY_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, py[N], SC, dt, n, λ0=λ0, f_str=f_str),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["N"],data["spike_counts"],λ0)   
    
end

function PY_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT,
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        py::Vector{Vector{TT}},spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
        λ0::Vector{Vector{UU}}=Vector{Vector{UU}}(),
        f_str::String="softplus") where {UU,TT <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #construct T x N spike count array
    spike_counts = hcat(spike_counts...)

    PS = Array{TT,2}(undef,n,T)
    c = Vector{TT}(undef,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks
    
    #construct T x N mean firing rate array
    λ0 = hcat(λ0...)

    @inbounds for t = 1:T

        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)
        y = hcat(map((py,c)-> fy2(py,xc,c, f_str=f_str), py, λ0[t,:])...)
        
        P .*= vec(exp.(sum(poiss_LL.(spike_counts[t,:], transpose(y), dt), dims=1)))
        
        PS[:,t] = P
        c[t] = sum(P)
        P /= c[t]

    end

    return PS

end

function P_all_trials(pz::Vector{TT}, data::Dict; 
        dt::Float64=1e-2, n::Int=53) where {TT <: Any}
        
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
                        
    output = pmap((L,R,T,nL,nR) -> P_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, dt, n), data["leftbups"], data["rightbups"], 
        data["nT"], data["binned_leftbups"], data["binned_rightbups"])   
    
end

function P_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT,
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        dt::Float64,n::Int) where {UU,TT <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    PS = Array{TT,2}(undef,n,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T

        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)
        PS[:,t] = P

    end

    return PS

end

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

function fy2(p::Vector{T},x::Vector{U},c::Float64;f_str::String="softplus") where {T,U <: Any}
    
    if f_str == "sig"

        y = exp.(p[3] .* x .+ p[4] .+ c)
        y[y .< 1e-150] .= p[1] + p[2]
        y[y .>= 1e150] .= p[1]
        y[(y .>= 1e-150) .& (y .< 1e150)] = p[1] .+ p[2] ./ (1. .+ y[(y .>= 1e-150) .& (y .< 1e150)])
        
    elseif f_str == "softplus"
        
        y = exp.(p[2] .* x .+ p[3] .+ c)
        y[y .< 1e-150] .= eps() + p[1]
        y[y .>= 1e150] .= 1e150
        y[(y .>= 1e-150) .& (y .< 1e150)] = (eps() + p[1]) .+ log.(1. .+ y[(y .>= 1e-150) .& (y .< 1e150)])
        
    end

    return y
    
end

"""
    poiss_LL(k,λ,dt)

    returns poiss LL
"""
function poiss_LL(k,λ,dt)
    
    #changed 2/17 to keep NaNs from gradient
    #if (λ*dt <= 1e-150) & (k == 0)  
    #if (λ*dt <= 1e-150)  
    #    k*log(1e-150) - λ*dt - lgamma(k+1)
        
    #else        
        k*log(λ*dt) - λ*dt - lgamma(k+1)
        
    #end
    
end

neural_null(k,λ,dt) = sum(poiss_LL.(k,λ,dt))

#=

function posterior_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT,
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        lambday::Array{TT,2}, spike_counts::Vector{Vector{Int}},dt::Float64,n::Int;
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

=#

########################## Model with RBF #################################################################

#=

function LL_all_trials(pz::Vector{TT}, py::Vector{Vector{TT}}, pRBF::Vector{Vector{TT}},
        data::Dict; dt::Float64=1e-2, n::Int=53,
        f_str::String="softplus", comp_posterior::Bool=false,
        numRBF::Int=20) where {TT <: Any}

    P,M,xc,dx, = initialize_latent_model(pz,n,dt)

    λ = hcat(fy.(py,[xc],f_str=f_str)...)
    #c = map(x->dt:dt:maximum(data["nT"][x])*dt,data["trial"])
    #rbf = map(x->UniformRBFE(x,numRBF),c);
    #λ0 = map((x,y,z)->x(y)*z, rbf, c, pRBF)

    λ0 = λ0_from_RBFs(pRBF,data;dt=dt,numRBF=numRBF)

    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, λ[:,N], SC, dt, n, λ0=λ0[N]),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"], data["N"],data["spike_counts"])

end

function λ0_from_RBFs(pRBF::Vector{Vector{TT}},data::Dict;
        dt::Float64=1e-2,numRBF::Int=20) where {TT <: Any}

    c = map(x->dt:dt:maximum(data["nT"][x])*dt,data["trial"])
    rbf = map(x->UniformRBFE(x,numRBF),c);
    λ0 = map((x,y,z)->x(y)*z, rbf, c, pRBF)

end
=#

#=

function LL_all_trials_old(pz::Vector{TT},py::Vector{Vector{TT}},
    data::Dict; dt::Float64=1e-2, n::Int=53, f_str::String="softplus", comp_posterior::Bool=false,
    λ0::Vector{Vector{Float64}}=Vector{Vector{Float64}}()) where {TT <: Any}

    P,M,xc,dx, = initialize_latent_model(pz,n,dt)

    λ = hcat(fy.(py,[xc],f_str=f_str)...)

    output = pmap((L,R,T,nL,nR,N,SC) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, λ[:,N], SC, dt, n, λ0=λ0[N]),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"], data["N"],data["spike_counts"])

end

=#
