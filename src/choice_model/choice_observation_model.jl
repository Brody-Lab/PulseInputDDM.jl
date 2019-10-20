"""
    bounded_mass_all_trials(pz, pd, data; dx=0.25)

    Computes the mass in the absorbing bin at the end of the trial consistent with the animal's choice. W
    Written for Diksha.

"""
function bounded_mass_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.25) where {TT}

    bias,lapse = pd[1],pd[2]
    dt = data["dt"]
    P,M,xc,n = initialize_latent_model(pz, dx, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    output = pmap((L,R,nT,nL,nR) -> P_single_trial(pz, P, M, dx, xc,
        L, R, nT, nL, nR, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"])

    return map((P,pokedR)-> (pokedR ? P[1] : P[n]), output, data["pokedR"])

end




"""
    LL_all_trials(pz, pd, data; dx=0.25)

"""
function LL_all_trials(pz::Vector{TT}, pd::Vector{UU}, data::Dict; dx::Float64=0.25) where {TT, UU <: Any}

    bias,lapse = pd[1],pd[2]
    dt = data["dt"]
    P,M,xc,n = initialize_latent_model(pz, dx, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    output = pmap((L,R,nT,nL,nR,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, nT, nL, nR, choice, bias, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"], data["pokedR"])

end




"""
    LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{VV}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        pokedR::Bool, bias::TT,
        n::Int, dt::Float64) where {TT,UU,VV <: Any}

"""
function LL_single_trial(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{VV}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        pokedR::Bool, bias::TT,
        n::Int, dt::Float64) where {TT,UU,VV <: Any}

    P = P_single_trial!(pz, P, M, dx, xc, L, R, nT, nL, nR, n, dt)
    P = likelihood!(bias, xc, P, pokedR, n, dx)

    return log(sum(P))

end



"""
    P_single_trial!(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{VV}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        n::Int, dt::Float64) where {TT,UU,VV <: Any}

"""
function P_single_trial!(pz::Vector{TT}, P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{VV}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        n::Int, dt::Float64) where {TT,UU,VV <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(pz,L,R)

    F = zeros(TT,n,n)     #empty transition matrix for time bins with clicks

    @inbounds for t = 1:nT

        P,F = latent_one_step!(P,F,pz,t,nL,nR,La,Ra,M,dx,xc,n,dt)

    end

    return P

end



"""
    ceil_and_floor(xc, s, n, dx)

"""
function ceil_and_floor(xc, s, n, dx)

    hp, lp = ceil(Int, (s-xc[2])/dx)+2, floor(Int, (s-xc[2])/dx)+2

    (hp < 1) && (hp = 1)
    (lp < 1) && (lp = 1)
    (hp > n) && (hp = n)
    (lp > n) && (lp = n)
    ((xc[1]<s) & (s<xc[2])) && (hp = 2)
    ((xc[end-1]<s) & (s<xc[end])) && (lp = n - 1)
    (xc[end] < s) && (hp = n; lp = n)
    (s < xc[1]) && (hp = 1; lp = 1)

    return hp, lp

end



"""
    likelihood!(bias::TT, xc, P, pokedR::Bool, n, dx) where {TT <: Any}

"""
function likelihood!(bias::TT, xc, P, pokedR::Bool, n, dx) where {TT <: Any}

    if ((bias > xc[end]) & (pokedR==true)) || ((bias < xc[1]) & (pokedR==false))
       P .= zero(TT)
    else

        hp, lp = ceil_and_floor(xc, bias, n, dx)

        if pokedR
            P[1:lp-1] .= zero(TT)
        else
            P[hp+1:end] .= zero(TT)
        end

        if lp==hp
            
            P[lp] = P[lp]/2
            
        else
            
            dh = xc[hp] - bias
            dl = bias - xc[lp]
            dd = dh + dl
            
            if pokedR
                P[hp] = P[hp] * (1/2 + dh/dd/2)
                P[lp] = P[lp] * (dh/dd/2)
            else
                P[hp] = P[hp] * (dl/dd/2)
                P[lp] = P[lp] * (1/2 + dl/dd/2)
            end
            
        end
    end

    return P

end



"""
    choice_null(choices)

"""
choice_null(choices) = sum(choices .== true)*log(sum(choices .== true)/length(choices)) +
    sum(choices .== false)*log(sum(choices .== false)/length(choices))

#=

    @inbounds for t = 1:T

        P,F = latent_one_step!(P,F,pz,t,hereL,hereR,La,Ra,M,dx,xc,n,dt)
        (t == T) && (P .*=  Pd)
        c[t] = sum(P)
        P /= c[t]
        comp_posterior ? post[:,t] = P : nothing

    end

    P = ones(Float64,n); #initialze backward pass with all 1's
    post[:,T] .*= P;

    @inbounds for t = T-1:-1:1

        (t + 1 == T) && (P .*=  Pd)
        P,F = latent_one_step!(P,F,pz,t+1,hereL,hereR,La,Ra,M,dx,xc,n,dt;backwards=true)
        P /= c[t+1]
        post[:,t] .*= P

    end

=#
