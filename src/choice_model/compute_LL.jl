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
    σ2_i, B, B_λ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, η, α_prior, β_prior, B_0, γ_shapeL, γ_scaleL, γ_shapeR, γ_scaleR = pz

    # computing initial values here
    a_0 = compute_initial_value(data, η, α_prior, β_prior)

    # adding the bias to a_0
    a_0 = a_0 .+ B_0

    L, R, nT, nL, nR, choice = data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"], data["pokedR"]
    dt = data["dt"]

    P = pmap((L,R,nT,nL,nR,a_0) -> P_single_trial!(σ2_i, B, B_λ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, lapse, γ_shapeL, γ_scaleL, γ_shapeR, γ_scaleR,
            L, R, nT, nL, nR, a_0, n, dt), L, R, nT, nL, nR, a_0)

    # For the new likelihood =======
    return log.(eps() .+ map((P, choice) -> (choice ? P[2] : P[1]), P, choice))
end


"""
    LL_all_trials(pz, pd, data; n=53)

Computes the log likelihood for a set of trials consistent with the animal's choice on each trial.

### Examples

```jldoctest
julia> pz, pd, data = default_parameters_and_data(generative=true, ntrials=10, rng=1);

julia> round.(LL_all_trials(pz["generative"], pd["generative"], data), digits=2)
10-element Array{Float64,1}:
 -0.23
 -0.04
 -0.04
 -0.15
 -0.03
 -0.14
 -0.04
 -0.03
 -0.03
 -0.15
```
"""
function LL_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT <: Any}

    if RTfit == true

        bounded_mass_all_trials(pz, pd, data; n=n)
    
    else
        bias, lapse = pd
        σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
        L, R, nT, nL, nR, choice = [data[key] for key in ["leftbups","rightbups","nT","binned_leftbups","binned_rightbups","pokedR"]]
        dt = data["dt"]

        P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, L_lapse=lapse/2, R_lapse=lapse/2)

        pmap((L,R,nT,nL,nR,choice) -> LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
            P, M, dx, xc, L, R, nT, nL, nR, choice, bias, n, dt), L, R, nT, nL, nR, choice)
    end
end


# """
# """
# function LL_all_trials(σ2_i::TT, B::TT, λ::TT, σ2_a::TT, σ2_s::TT,
#     ϕ::TT, τ_ϕ::TT, bias::TT, lapse::TT, data::Dict; n::Int=53) where {TT <: Any}

#     if RTfit == true
    
#         error("yet to be implemented, can not call with these arguments")
    
#     else

#         L, R, nT, nL, nR, choice = [data[key] for key in ["leftbups","rightbups","nT","binned_leftbups","binned_rightbups","pokedR"]]
#         dt = data["dt"]

#         P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, L_lapse=lapse/2, R_lapse=lapse/2)

#         pmap((L,R,nT,nL,nR,choice) -> LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
#             P, M, dx, xc, L, R, nT, nL, nR, choice, bias, n, dt), L, R, nT, nL, nR, choice)
#     end
# end


# """
#     LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
#         P, M, dx, xc, L, R, nT, nL, nR, pokedR bias, n, dt)
# """
# function LL_single_trial!(λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT,
#         P::Vector{TT}, M::Array{TT,2}, dx::UU,
#         xc::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
#         nL::Vector{Int}, nR::Vector{Int},
#         pokedR::Bool, bias::TT,
#         n::Int, dt::Float64) where {TT,UU <: Any}

#     if RTfit == true

#         error("yet to be implemented, can not call with these arguments")
    
#     else

#         P = P_single_trial!(λ,σ2_a,σ2_s,ϕ,τ_ϕ,P,M,dx,xc,L,R,nT,nL,nR,n,dt)
#         P = choice_likelihood!(bias,xc,P,pokedR,n,dx)

#         return log(sum(P))

#     end

# end





"""
    P_single_trial!(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, lapse
             L, R, nT, nL, nR, a_0, n, dt)

"""
function P_single_trial!(σ2_i::TT, B::TT, B_λ::TT, λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT, lapse::TT, γ_shapeL::TT, γ_scaleL::TT, γ_shapeR::TT, γ_scaleR::TT,
        L::Vector{Float64}, R::Vector{Float64}, nT::Int, nL::Vector{Int}, nR::Vector{Int}, a_0::TT,
        n::Int, dt::Float64) where {TT <: Any}

    absB = 1

    P,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, a_0, absB, L_lapse=lapse/2, R_lapse=lapse/2)

    # computing sticky bounds for each time step -- this will fail if the collapse is really high 
    Bt = zeros(TT,nT)
    numsticky = zeros(TT,nT)
    tv = collect(1:nT)
    Bt =  B .* exp.((B_λ*dt).*tv)
    numsticky = absB .+ floor.(Int, cumsum(abs.(diff(Bt))./dx))
    numsticky = [numsticky; numsticky[end]]

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(ϕ,τ_ϕ,L,R)

    # empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)

    # For the new likelihood =======
    Pbounds = zeros(TT, 2, nT)

    
    @inbounds for t = 1:nT

        absB = floor.(Int, numsticky[t])

        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,dx,xc,n,dt,absB)

        # For the new likelihood =======
        if t == 1
            Pbounds[1,t] = sum(P[1:absB])  # left bound
            Pbounds[2,t] = sum(P[end-(absB-1):end])  # right bound
        else 
            Pbounds[1,t] = sum(P[1:absB]) - sum(Pbounds[1, :])   # mass differential
            Pbounds[2,t] = sum(P[end-(absB-1):end])  - sum(Pbounds[2, :])   # mass differential
        end

    end
    
    if RTfit == true
        
        # For the new likelihood =======
        NDtimedistL = Gamma(γ_shapeL, γ_scaleL)
        NDtimedistR = Gamma(γ_shapeR, γ_scaleR)

        tvec = dt .* collect(nT:-1:1)

        pback = zeros(TT,2)
        pback[1] = transpose(Pbounds[1,:]) * (pdf.(NDtimedistL, tvec) .* dt)
        pback[2] = transpose(Pbounds[2,:]) * (pdf.(NDtimedistR, tvec) .* dt)
        return pback

    else
        return P
    end
end




# """
#     choice_likelihood!(bias, xc, P, pokedR, n, dx)

# Preserves mass in the distribution P on the side consistent with the choice pokedR relative to the point bias. Deals gracefully in situations where the bias equals a bin center. However, if the bias grows larger than the bound, the LL becomes very large and the gradient is zero. However, it's general convexity of the -LL surface w.r.t this parameter should generally preclude it from approaches these regions.

# ### Examples

# ```jldoctest
# julia> n, dt = 13, 1e-2;

# julia> bias = 0.51;

# julia> σ2_i, B, λ, σ2_a = 1., 2., 0., 10.; # the bound height of 2 is intentionally low, so P is not too long

# julia> P, M, xc, dx = pulse_input_DDM.initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt);

# julia> pokedR = true;

# julia> round.(pulse_input_DDM.choice_likelihood!(bias, xc, P, pokedR, n, dx), digits=2)
# 13-element Array{Float64,1}:
#  0.0
#  0.0
#  0.0
#  0.0
#  0.0
#  0.0
#  0.0
#  0.04
#  0.09
#  0.08
#  0.05
#  0.03
#  0.02
# ```
# """
# function choice_likelihood!(bias::TT, xc::Vector{TT}, P::Vector{TT},
#                  pokedR::Bool, n::Int, dx::UU) where {TT,UU <: Any}

#     lp = searchsortedlast(xc,bias)
#     hp = lp + 1

#     if ((hp==n+1) & (pokedR==true))
#         P[1:lp-1] .= zero(TT)
#         P[lp] = eps()

#     elseif((lp==0) & (pokedR==false))
#         P[hp+1:end] .= zero(TT)
#         P[hp] = eps()

#     elseif ((hp==n+1) & (pokedR==false)) || ((lp==0) & (pokedR==true))
#         P .= one(TT)

#     else

#         dh, dl = xc[hp] - bias, bias - xc[lp]
#         dd = dh + dl

#         if pokedR
#             P[1:lp-1] .= zero(TT)
#             P[hp] = P[hp] * (1/2 + dh/dd/2)
#             P[lp] = P[lp] * (dh/dd/2)
#         else
#             P[hp+1:end] .= zero(TT)
#             P[hp] = P[hp] * (dl/dd/2)
#             P[lp] = P[lp] * (1/2 + dl/dd/2)
#         end

#     end

#     return P

# end


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
