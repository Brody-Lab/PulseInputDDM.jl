"""
"""
function loglikelihood(θ::θchoice, data; n::Int=53)

    @unpack θz, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    sum(pmap(data -> loglikelihood!(θ, P, M, dx, xc, data, n), data))

end


"""
"""
(θ::θchoice)(data::Vector{choicedata}; n::Int=53) = loglikelihood(θ, data; n=n)


"""
    loglikelihood!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, pokedR bias, n, dt)
"""
function loglikelihood!(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data::choicedata,
        n::Int) where {TT,UU <: Real}

    @unpack θz, bias = θ
    @unpack click_data, choice = data

    P = P_single_trial!(θz,P,M,dx,xc,click_data,n)
    log(sum(choice_likelihood!(bias,xc,P,choice,n,dx)))

end


"""
    P_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, n, dt)

"""
function P_single_trial!(θz,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, click_data,
        n::Int) where {TT,UU <: Real}

    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R)

    #empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)

    @inbounds for t = 1:nT

        #maybe only pass one L,R,nT?
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)

    end

    return P

end


"""
    choice_likelihood!(bias, xc, P, pokedR, n, dx)

Preserves mass in the distribution P on the side consistent with the choice pokedR relative to the point bias. Deals gracefully in situations where the bias equals a bin center. However, if the bias grows larger than the bound, the LL becomes very large and the gradient is zero. However, it's general convexity of the -LL surface w.r.t this parameter should generally preclude it from approaches these regions.

### Examples

```jldoctest
julia> n, dt = 13, 1e-2;

julia> bias = 0.51;

julia> σ2_i, B, λ, σ2_a = 1., 2., 0., 10.; # the bound height of 2 is intentionally low, so P is not too long

julia> P, M, xc, dx = pulse_input_DDM.initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt);

julia> pokedR = true;

julia> round.(pulse_input_DDM.choice_likelihood!(bias, xc, P, pokedR, n, dx), digits=2)
13-element Array{Float64,1}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.04
 0.09
 0.08
 0.05
 0.03
 0.02
```
"""
function choice_likelihood!(bias::TT, xc::Vector{TT}, P::Vector{TT},
                 pokedR::Bool, n::Int, dx::UU) where {TT,UU <: Any}

    lp = searchsortedlast(xc,bias)
    hp = lp + 1

    if ((hp==n+1) & (pokedR==true))
        P[1:lp-1] .= zero(TT)
        P[lp] = eps()

    elseif((lp==0) & (pokedR==false))
        P[hp+1:end] .= zero(TT)
        P[hp] = eps()

    elseif ((hp==n+1) & (pokedR==false)) || ((lp==0) & (pokedR==true))
        P .= one(TT)

    else

        dh, dl = xc[hp] - bias, bias - xc[lp]
        dd = dh + dl

        if pokedR
            P[1:lp-1] .= zero(TT)
            P[hp] = P[hp] * (1/2 + dh/dd/2)
            P[lp] = P[lp] * (dh/dd/2)
        else
            P[hp+1:end] .= zero(TT)
            P[hp] = P[hp] * (dl/dd/2)
            P[lp] = P[lp] * (1/2 + dl/dd/2)
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

    """
        bounded_mass_all_trials(pz, pd, data; n=53)

    Computes the mass in the absorbing bin at the end of the trial consistent with the animal's choice.

    ### Examples
    ```jldoctest
    julia> pz, pd, data = default_parameters_and_data(generative=true, ntrials=10, rng=1);

    julia> round.(bounded_mass_all_trials(pz["generative"], pd["generative"], data), digits=2)
    10-element Array{Float64,1}:
     0.04
     0.4
     0.2
     0.04
     0.53
     0.06
     0.45
     0.06
     0.63
     0.15
    ```
    """
    function bounded_mass_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT}

        bias, lapse = pd
        σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
        L, R, nT, nL, nR, choice = data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
            data["binned_rightbups"], data["pokedR"]
        dt = data["dt"]

        P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, L_lapse=lapse/2, R_lapse=lapse/2)

        P = pmap((L,R,nT,nL,nR) -> P_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
            P, M, dx, xc, L, R, nT, nL, nR, n, dt), L, R, nT, nL, nR)

        return map((P,choice)-> (choice ? P[n] : P[1]), P, choice)

    end


=#
