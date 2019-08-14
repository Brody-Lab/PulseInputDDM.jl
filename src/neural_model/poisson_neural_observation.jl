
function LL_all_trials(pz::Vector{TT}, py::Vector{Vector{UU}}, data::Dict, 
        n::Int, f_str::String) where {TT,UU <: Any}
     
    dt = data["dt"]
    use_bin_center = data["use_bin_center"] #this should always be false for this model
    P,M,xc,dx, = initialize_latent_model(pz,n,dt)
    
    #trials = sample(1:data["ntrials"], min(100,data["ntrials"]), replace = false)
    trials = sample(1:data["ntrials"], data["ntrials"], replace = false)
                            
    output = pmap((L,R,T,nL,nR,SC,λ0) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, py, SC, dt, n, λ0, f_str, use_bin_center),
        data["leftbups"][trials], data["rightbups"][trials], data["nT"][trials], 
        data["binned_leftbups"][trials], 
        data["binned_rightbups"][trials], data["spike_counts"][trials], 
        data["λ0"][trials], batch_size=1)   
    
end

function LL_single_trial(pz::Vector{ZZ}, P::Vector{TT}, M::Array{TT,2}, dx::VV,
        xc::Vector{WW},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        nL::Vector{Int}, nR::Vector{Int},
        py::Vector{Vector{YY}}, k::Vector{Vector{Int}},dt::Float64,n::Int,
        λ0::Vector{Vector{UU}},
        f_str::String, use_bin_center::Bool) where {UU,TT,VV,WW,YY,ZZ <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    c = Vector{TT}(undef,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T

        if use_bin_center && t == 1
            P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt/2)
        else
            P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        end
        
        P .*= vcat(map(xc-> exp(sum(map((k,py,λ0)-> logpdf(Poisson(f_py(xc,λ0[t],py,f_str) * dt), 
                                k[t]), k, py, λ0))), xc)...)
        
        c[t] = sum(P)
        P /= c[t]

    end

    return sum(log.(c))

end

function f_py(x::U, c::Float64, p::Vector{T}, f_str::String) where {T,U <: Any}

    if f_str == "sig"
        
        #y = p[3] * x + p[4] + log(c)             
        #y = p[1] + p[2] * logistic!(y)
        
        y = p[3] * x + p[4]        
        y = p[1] + p[2] * logistic!(y)
        #y = softplus(y + c)
        y = max(eps(),y+c)
        
    elseif f_str == "softplus"
        
        #y = p[1] + log(1. + exp((p[2] * x + p[3]) + c))
        y = p[1] + softplus(p[2]*x + p[3] + c)
        
    end

    return y
    
end

function f_py!(x::U, c::Float64, p::Vector{T}, f_str::String) where {T,U <: Any}

    if f_str == "sig"
        
        #x = p[3] * x + p[4] + log(c)       
        #x = p[1] + p[2] * logistic!(x)
        
        x = p[3] * x + p[4]      
        x = p[1] + p[2] * logistic!(x)
        #x = softplus(x + c)
        x = max(eps(),x+c)
        
    elseif f_str == "softplus"
        
        #x = p[1] + log(1. + exp(p[2] * x + p[3] + c))
        x = p[1] + softplus(p[2]*x + p[3] + c)
        
    end

    return x
    
end

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

neural_null(k,λ,dt) = sum(logpdf.(Poisson.(λ*dt),k))

#=

function LL_all_trials_dx(pz::Vector{TT}, py::Vector{Vector{TT}}, data::Dict, 
        dx::Float64, f_str) where {TT <: Any}
     
    dt = data["dt"]
    P,M,xc,n, = initialize_latent_model_dx(pz,dx,dt)
                            
    output = pmap((L,R,T,nL,nR,SC,λ0) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, T, nL, nR, py, SC, dt, n, λ0, f_str),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["spike_counts"], data["λ0"])   
    
end

#for testing
function LL_single_trial_dx(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::VV,
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        nL::Vector{Int}, nR::Vector{Int},
        py::Vector{Vector{TT}}, k::Vector{Vector{Int}},dt::Float64,n::Int,
        λ0::Vector{Vector{UU}},
        f_str::String) where {UU,TT,VV <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    c = Vector{TT}(undef,T)
    #PS = Array{TT,2}(undef,n,T)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks
    
    #construct T x N mean firing rate array and spike count array
    #λ0 = hcat(λ0...)
    #k = hcat(k...)

    @inbounds for t = 1:T

        P, = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        P .*= vcat(map(xc-> exp(sum(map((k,py,λ0)-> logpdf(Poisson(f_py(xc,λ0[t],py,f_str) * dt), 
                                k[t]), k, py, λ0))), xc)...)
        
        c[t] = sum(P)
        #PS[:,t] = P
        P /= c[t]

    end

    #return PS, c #sum(log.(c))
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
