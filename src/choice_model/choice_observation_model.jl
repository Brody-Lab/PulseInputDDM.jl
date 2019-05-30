
function LL_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT}
        
    bias,lapse = pd[1],pd[2]
    dt = data["dt"]
    P,M,xc,dx,xe = initialize_latent_model(pz, n,dt, L_lapse=lapse/2, R_lapse=lapse/2)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
            
    output = pmap((L,R,nT,nL,nR,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, nT, nL, nR, nbinsL, Sfrac, choice, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["pokedR"])   
    
end

function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::UU, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool,
        n::Int, dt::Float64) where {TT,UU <: Any}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
        
    F = zeros(TT,n,n)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:nT
        
        P,F = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)               
        (t == nT) && (P .*=  Pd)

    end

    return log(sum(P))

end

function bias_bin(bias::TT,xe::Vector{TT},dx::UU,n::Int) where {TT,UU <: Any}
    
    nbinsL = sum(bias .> xe[2:n])
    Sfrac = (bias - xe[nbinsL+1])/dx
    Sfrac < zero(Sfrac) ? Sfrac = zero(Sfrac) : nothing
    Sfrac > one(Sfrac) ? Sfrac = one(Sfrac) : nothing
    
    return nbinsL, Sfrac
    
end

choice_null(choices) = sum(choices .== true)*log(sum(choices .== true)/length(choices)) + 
    sum(choices .== false)*log(sum(choices .== false)/length(choices))

#################################### New choice model #################################


function LL_all_trials_new(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT}
        
    dt = data["dt"]
    P,M,xc,dx, = initialize_latent_model(pz, n, dt)
            
    output = pmap((L,R,nT,nL,nR,choice) -> LL_single_trial(pz, pd, P, M, dx, xc,
        L, R, nT, nL, nR, choice, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"], data["pokedR"])   
    
end

function LL_single_trial(pz::Vector{TT}, pd::Vector{TT},
        P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        pokedR::Bool, n::Int, dt::Float64) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)
            
    F = zeros(TT,n,n)
    
    @inbounds for t = 1:nT
        
        P,F = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)               
        (t == nT) && (P .*=  vcat(map(x-> exp(Bern_LL(Bern_sig(pd,x),pokedR)), xc)...))

    end

    return log(sum(P))

end

Bern_LL(p,x) = log(p)*x + log(1-p)*(1. -x)
Bern_sig(p,x) = p[1] + (p[2] - p[1]) * logistic((-p[3] * x) - p[4])

#=
#this is outdated and won't work, but want to keep around until I fix it
function LL_single_trial_w_posterior(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::TT, 
        xc::Vector{TT},L::Vector{Float64}, R::Vector{Float64}, T::Int,
        hereL::Vector{Int}, hereR::Vector{Int},
        nbinsL::Union{TT,Int}, Sfrac::TT, pokedR::Bool,
        n::Int, dt::Float64;
        comp_posterior::Bool=false) where {TT}
    
    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    #vector to sum choice evidence
    pokedL = convert(TT,!pokedR); pokedR = convert(TT,pokedR)
    Pd = vcat(pokedL * ones(nbinsL), pokedL * Sfrac + pokedR * (one(Sfrac) - Sfrac), pokedR * ones(n - (nbinsL + 1)))
        
    c = Vector{TT}(undef,T)
    comp_posterior ? post = Array{Float64,2}(unde,n,T) : nothing
    F = zeros(TT,n,n)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:T
        
        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)               
        (t == T) && (P .*=  Pd)
        c[t] = sum(P)
        P /= c[t] 
        comp_posterior ? post[:,t] = P : nothing

    end

    if comp_posterior

        P = ones(Float64,n); #initialze backward pass with all 1's  
        post[:,T] .*= P;

        @inbounds for t = T-1:-1:1
            
            (t + 1 == T) && (P .*=  Pd)
            P,F = latent_one_step!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
            P /= c[t+1] 
            post[:,t] .*= P

        end

    end

    comp_posterior ? (return post) : (return sum(log.(c)))

end

=#