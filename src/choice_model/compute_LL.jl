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
function LL_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.1) where {TT <: Any}

    bias, lapse = pd
    σ2_i, B, B_λ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, η, α_prior, β_prior, B_0, γ_shape, γ_scale, γ_shape1, γ_scale1 = pz

    # computing initial values here
    a_0 = compute_initial_value(data, η, α_prior, β_prior)

    # adding the bias to a_0
    a_0 = a_0 .+ B_0

    L, R, nT, nL, nR, choice = data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
        data["binned_rightbups"], data["pokedR"]
    dt = data["dt"]

    P = pmap((L,R,nT,nL,nR,a_0) -> P_single_trial!(σ2_i, B, B_λ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, lapse, γ_shape, γ_scale, γ_shape1, γ_scale1,
            L, R, nT, nL, nR, a_0, dx, dt), L, R, nT, nL, nR, a_0)
    
    return log.(eps() .+ map((P, choice) -> (choice ? P[2] : P[1]), P, choice))
end



"""
    P_single_trial!(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, lapse
             L, R, nT, nL, nR, a_0, n, dt)

"""
function P_single_trial!(σ2_i::TT, B::TT, B_λ::TT, λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT, lapse::TT, γ_shape::TT, γ_scale::TT, γ_shape1::TT, γ_scale1::TT,
        L::Vector{Float64}, R::Vector{Float64}, nT::Int, nL::Vector{Int}, nR::Vector{Int}, a_0::TT,
        dx::Float64, dt::Float64) where {TT <: Any}

    # computing the bound
    Bt = zeros(TT,nT)
    Bt = map(x-> sqrt(B_λ+x)*sqrt(2)*erfinv(2*B - 1.), dt .* collect(1:nT))

    #adapt magnitude of the click inputs
    La, Ra = make_adapted_clicks(ϕ,τ_ϕ,L,R)

    # initialize latent model with a_0
    P, xc, n = initialize_latent_model(σ2_i, Bt[1], λ, σ2_a, dx, dt, a_0,L_lapse=lapse/2, R_lapse=lapse/2)

    F = zeros(TT,n,n)
    
    # to keep track of mass at the bounds  
    Pbounds = zeros(TT, 2, nT)

    
    @inbounds for t = 1:nT

        xc,n = bins(Bt[t],xc)    
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,dx,xc,n,dt)

        if t == 1
            Pbounds[1,t] = P[1]  # left bound
            Pbounds[2,t] = P[n]  # right bound
        else 
            Pbounds[1,t] = P[1] - sum(Pbounds[1, :])   # mass differential
            Pbounds[2,t] = P[n] - sum(Pbounds[2, :])   # mass differential
        end

    end
    
    # multiplying mass at the bounds with prob of observing the remaining NDtime
    if RTfit == true
      
        NDtimedistL = Gamma(γ_shape, γ_scale)
        NDtimedistR = Gamma(γ_shape1, γ_scale1)

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
#     P_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
#         P, M, dx, xc, L, R, nT, nL, nR, n, dt)

# """
# function P_single_trial!(λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT,
#         P::Vector{TT}, M::Array{TT,2}, dx::UU,
#         xc::Vector{TT}, L::Vector{Float64}, R::Vector{Float64}, nT::Int,
#         nL::Vector{Int}, nR::Vector{Int},
#         n::Int, dt::Float64) where {TT,UU <: Any}

#     #adapt magnitude of the click inputs
#     La, Ra = make_adapted_clicks(ϕ,τ_ϕ,L,R)

#     #empty transition matrix for time bins with clicks
#     F = zeros(TT,n,n)
#     Pt_1 = zeros(TT,n)
#     Ft_1 = zeros(TT,n,n)
    
#     @inbounds for t = 1:nT
    
#         if t == nT-1    
#             Pt_1, Ft_1 = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
#         end

#         P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
    
#     end
    
#     if RTfit == true
#         return (P - Pt_1)   # getting the mass that hits the bound at the very last time step
#     else
#         return P
#     end
# end




"""
    choice_null(choices)

"""
choice_null(choices) = sum(choices .== true)*log(sum(choices .== true)/length(choices)) +
    sum(choices .== false)*log(sum(choices .== false)/length(choices))


