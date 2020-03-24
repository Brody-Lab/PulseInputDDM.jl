"""
"""
function logistic!(x::T) where {T <: Any}

    if x >= 0.
        x = exp(-x)
        x = 1. / (1. + x)
    else
        x = exp(x)
        x = x / (1. + x)
    end

    return x

end


"""
"""
neural_null(k,λ,dt) = sum(logpdf.(Poisson.(λ*dt),k))

#=

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

function LL_all_trials_threads(pz::Vector{TT}, py::Vector{Vector{TT}}, data::Dict,
        n::Int, f_str::String) where {TT <: Any}

    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    trials = length(data["nT"])
    LL = Vector{TT}(undef,trials)

    @threads for i = 1:length(data["nT"])
        LL[i] = LL_single_trial(pz, copy(P), M, dx, xc,
                data["leftbups"][i], data["rightbups"][i], data["nT"][i],
                data["binned_leftbups"][i], data["binned_rightbups"][i], py,
                data["spike_counts"][i], dt, n, data["λ0"][i], f_str)
    end

    return LL

end

=#
