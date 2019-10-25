"""
    bounded_mass_all_trials(pz, pd, data; dx=0.25)

Computes the mass in the absorbing bin at the end of the trial consistent with the animal's choice.

### Examples
```jldoctest
julia> pz, pd = default_parameters(generative=true);

julia> ntrials, use_bin_center, rng = 10, false, 1;

julia> data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng);

julia> data = bin_clicks!(data,use_bin_center);

julia> bounded_mass_all_trials(pz["generative"], pd["generative"], data)
10-element Array{Float64,1}:
 0.4730234305246653  
 0.050000000000690735
 0.9498457918453798  
 0.6502294480995431  
 0.9498701314348695  
 0.9486683638140332  
 0.9499602489378979  
 0.9494364520875525  
 0.9497941680881434  
 0.857734284166645 
```
"""
function bounded_mass_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.25) where {TT}

    bias, lapse = pd
    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
    L, R, nT, nL, nR, choice = data["left"], data["right"], data["nT"], data["binned_left"],
        data["binned_right"], data["pokedR"]
    dt = data["dt"]
    
    P,M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    P = pmap((L,R,nT,nL,nR) -> P_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        P, M, dx, xc, L, R, nT, nL, nR, n, dt), L, R, nT, nL, nR)

    return map((P,choice)-> (choice ? P[n] : P[1]), P, choice)

end


"""
    LL_all_trials(pz, pd, data; dx=0.25)

Computes the log likelihood for a set of trials consistent with the animal's choice on each trial.

### Examples

```jldoctest
julia> pz, pd = default_parameters(generative=true);

julia> ntrials, use_bin_center, rng = 10, false, 1;

julia> data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials; rng=rng);

julia> data = bin_clicks!(data,use_bin_center);

julia> LL_all_trials(pz["generative"], pd["generative"], data)
10-element Array{Float64,1}:
 -0.09673131654173264 
 -2.995732195594889   
 -0.05129331779430949 
 -0.06204809221624061 
 -0.05129413814902191 
 -0.05129408701986206 
 -0.05129402919086469 
 -0.051293301292408694
 -0.051293585770327346
 -0.056150291769304285
```
"""
function LL_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.25) where {TT <: Any}

    bias, lapse = pd
    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
    L, R, nT, nL, nR, choice = [data[key] for key in ["left","right","nT","binned_left","binned_right","pokedR"]]
    dt = data["dt"]

    P,M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    pmap((L,R,nT,nL,nR,choice) -> LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, choice, bias, n, dt), L, R, nT, nL, nR, choice)

end


"""
    LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, pokedR bias, n, dt)
"""
function LL_single_trial!(λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT,
        P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{VV}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        pokedR::Bool, bias::TT,
        n::Int, dt::Float64) where {TT,UU,VV <: Any}

    P = P_single_trial!(λ,σ2_a,σ2_s,ϕ,τ_ϕ,P,M,dx,xc,L,R,nT,nL,nR,n,dt)
    P = choice_likelihood!(bias,xc,P,pokedR,n,dx)

    return log(sum(P))

end


"""
    P_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, n, dt)

"""
function P_single_trial!(λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT,
        P::Vector{TT}, M::Array{TT,2}, dx::Float64,
        xc::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
        nL::Vector{Int}, nR::Vector{Int},
        n::Int, dt::Float64) where {TT <: Any}

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(ϕ,τ_ϕ,L,R)

    #empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)

    @inbounds for t = 1:nT

        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)

    end

    return P

end


"""
    choice_likelihood!(bias, xc, P, pokedR, n, dx)

"""
function choice_likelihood!(bias::TT, xc::Vector{TT}, P::Vector{TT}, 
                 pokedR::Bool, n::Int, dx::Float64) where {TT <: Any}

    hp, lp = searchsortedfirst(xc,bias), searchsortedlast(xc,bias)

    if ((hp == n+1) & (pokedR==true)) || ((lp == 0) & (pokedR==false))

       P .= zero(TT)

    elseif ((hp == n+1) & (pokedR==false)) || ((lp == 0) & (pokedR==true))

       P .= one(TT)

    else

        if pokedR
            P[1:lp-1] .= zero(TT)
        else
            P[hp+1:end] .= zero(TT)
        end

        if lp==hp

            #P[lp] = P[lp]/2
            dh = xc[lp] - bias
            P[lp] = P[lp] * (1/2 + dh)

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
