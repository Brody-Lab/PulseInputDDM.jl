"""
    loglikelihood(model, n)
Given a model, computes the log likelihood for a set of trials.
```
"""
function loglikelihood(model::T, dx::Float64) where T <: DDM

    @unpack θ, data = model
    data_dict = make_data_dict(data)
    loglikelihood(θ, data, data_dict, dx)

end


"""
    loglikelihood(θ, data, n)
Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(θ::DDMθ, data, data_dict, dx::Float64)

    # initial point computation
    a_0 = compute_initial_pt(θ.hist_θz, θ.base_θz.bias, data_dict)  

    # non-decision time distribution computation
    @unpack ndtimeL1, ndtimeL2 = θ.ndtime_θz
    @unpack ndtimeR1, ndtimeR2 = θ.ndtime_θz
    NDdistL = Gamma(ndtimeL1, ndtimeL2)
    NDdistR = Gamma(ndtimeR1, ndtimeR2)

    dt = data_dict["dt"]
    frac = data_dict["frac"]

    P = pmap((data, a_0, nT) -> loglikelihood!(θ.base_θz, data, a_0, dx, pdf.(NDdistL, dt.*collect(nT:-1:1)).*dt, 
                                        pdf.(NDdistR, dt.*collect(nT:-1:1)).*dt), data, a_0, data_dict["nT"])
    
    return sum(log.((frac .* data_dict["lapse_lik"] .* .5) .+ (1. - frac)
        .*map((P, choice) -> (choice ? P[2] : P[1]), P, data_dict["choice"])))  

end

"""
    loglikelihood!(θ, data, i_0, n)
    
Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
function loglikelihood!(base_θz::θz_base,data::choicedata,
        a_0::TT, dx::Float64, ndL, ndR) where {TT <: Any}

    @unpack click_data, choice = data

    if base_θz.Bλ == 0
        Pbounds = P_single_trial!(base_θz, dx, click_data, a_0)
    else
        Pbounds = P_single_trial!(base_θz, dx, click_data, a_0, base_θz.Bλ)
    end

    # non-decision time 
    pback = zeros(TT,2)
    pback[1] = transpose(Pbounds[1,:]) * ndL
    pback[2] = transpose(Pbounds[2,:]) * ndR
        
    return pback

end


"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)
Given parameters θz progagates P for one trial
speed ups for when bound is stationary
"""
function P_single_trial!(base_θz::θz_base, dx::Float64,
        click_data, a_0::TT) where {TT<: Any}

    @unpack λ, σ2_i, σ2_a, bias, h_drift_scale = base_θz
    @unpack σ2_s, ϕ, τ_ϕ, B0 = base_θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks
   
    P, xc, n = initialize_latent_model(σ2_i, B0, λ, σ2_a, dx, dt, a_0 .+ bias)
    La, Ra   = adapt_clicks(ϕ,τ_ϕ,L,R)

    Pbounds = zeros(TT, 2, nT)
    F = zeros(TT, n, n)

    @inbounds for t = 1:nT

        P, F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,a_0*h_drift_scale,dx,xc,n,dt)

        if t == 1
            Pbounds[1,t] = P[1] # left bound
            Pbounds[2,t] = P[n] # right bound
        else
            Pbounds[1,t] = P[1] - sum(Pbounds[1,:]) # mass differential
            Pbounds[2,t] = P[n] - sum(Pbounds[2,:]) # mass differential
        end

    end

    return Pbounds

end



"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)
Given parameters θz progagates P for one trial
when bound is non stationary
"""
function P_single_trial!(base_θz::θz_base, dx::Float64,
        click_data, a_0::TT, Bλ::TT) where {TT<: Any}

    @unpack λ, σ2_i, σ2_a, bias, h_drift_scale = base_θz
    @unpack σ2_s, ϕ, τ_ϕ, B0, Bm = base_θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #computing the bound
    B = zeros(TT,nT)
    # Bt = map(x-> sqrt(B_λ+x)*sqrt(2)*erfinv(2*B - 1.), dt .* collect(1:nT))
    # Bt = map(x->B*(1. + exp(B_λ*(x-B_Δ)))^(-1.), dt .* collect(1:nT))
    B = map(x->B0 + Bλ*sqrt(x), dt .* collect(1:nT))

    P, xc, n = initialize_latent_model(σ2_i, B0, λ, σ2_a, dx, dt, a_0 .+ bias)
    La, Ra   = adapt_clicks(ϕ,τ_ϕ,L,R)

    Pbounds = zeros(TT, 2, nT)


    @inbounds for t = 1:nT

        n_pre = n
        xc_pre = xc
        xc,n = bins(B[t], dx)   

        F = zeros(TT,n,n_pre)

        P, F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,a_0*h_drift_scale,dx,xc,xc_pre,n,dt)

        if t == 1
            Pbounds[1,t] = P[1] # left bound
            Pbounds[2,t] = P[n] # right bound
        else
            Pbounds[1,t] = P[1] - sum(Pbounds[1,:]) # mass differential
            Pbounds[2,t] = P[n] - sum(Pbounds[2,:]) # mass differential
        end

    end

    return Pbounds

end



# function LL_all_trials(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.1) where {TT <: Any}

   

#     L, R, nT, nL, nR, choice = data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"],
#         data["binned_rightbups"], data["pokedR"]
#     dt = data["dt"]

#     P = pmap((L,R,nT,nL,nR,a_0) -> P_single_trial!(σ2_i, B, B_λ, B_Δ, λ, σ2_a, σ2_s, ϕ, τ_ϕ, γ_shape, γ_scale, γ_shape1, γ_scale1,
#             L, R, nT, nL, nR, a_0, dx, dt), L, R, nT, nL, nR, a_0)
    
#     # lapse, lapse1, lapse2 = pd

#     # NDtimedistL = Gamma(γ_shape, γ_scale)
#     # NDtimedistR = Gamma(γ_shape1, γ_scale1)
#     # lapse_lik = map((choice,nT) -> (choice ? pdf.(NDtimedistR, nT .* dt) : pdf.(NDtimedist1, nT .* dt) )) .* dt
#     # return log.(lapse .* lapse_lik .+ (1-lapse) .* map((P, choice) -> (choice ? P[2] : P[1]), P, choice))

#     frac = 1e-4;

#     lapse_dist = Exponential(0.1495)
#     lapse_lik = pdf.(lapse_dist,nT.*dt) .*dt
#     return log.(frac .* lapse_lik .* .5 .+ (1. - frac).*map((P, choice) -> (choice ? P[2] : P[1]), P, choice))  # for robustness


#     # return log.(frac .+ (1. - 2. * frac).*map((P, choice) -> (choice ? P[2] : P[1]), P, choice))  # for robustness



# end



# """
#     P_single_trial!(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, lapse
#              L, R, nT, nL, nR, a_0, n, dt)  

# """
# function P_single_trial!(σ2_i::TT, B::TT, B_λ::TT, B_Δ::TT, λ::TT, σ2_a::TT, σ2_s::TT, ϕ::TT, τ_ϕ::TT, γ_shape::TT, γ_scale::TT, γ_shape1::TT, γ_scale1::TT,
#         L::Vector{Float64}, R::Vector{Float64}, nT::Int, nL::Vector{Int}, nR::Vector{Int}, a_0::TT,
#         dx::Float64, dt::Float64) where {TT <: Any}

#     # computing the bound
#     Bt = zeros(TT,nT)
#     # Bt = map(x-> sqrt(B_λ+x)*sqrt(2)*erfinv(2*B - 1.), dt .* collect(1:nT))
#     # Bt = map(x->B*(1. + exp(B_λ*(x-B_Δ)))^(-1.), dt .* collect(1:nT))
#     Bt = map(x->B + B_λ*sqrt(x), dt .* collect(1:nT))

#     #adapt magnitude of the click inputs
#     La, Ra = make_adapted_clicks(ϕ,τ_ϕ,L,R)
#     # La, Ra = make_adapted_clicks(ϕ, L, R)

#     # initialize latent model with a_0
#     P, xc, n = initialize_latent_model(σ2_i, Bt[1], λ, σ2_a, dx, dt, a_0)

#     # to keep track of mass at the bounds  
#     Pbounds = zeros(TT, 2, nT)
    
#     @inbounds for t = 1:nT

#         n_pre = n
#         xc_pre = xc
#         xc,n = bins(Bt[t], dx)   

#         F = zeros(TT,n,n_pre)

#         P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,dx,xc,xc_pre,n,dt)

#         if t == 1
#             Pbounds[1,t] = P[1]  # left bound
#             Pbounds[2,t] = P[n]  # right bound
#         else 
#             Pbounds[1,t] = P[1] - sum(Pbounds[1, :])   # mass differential
#             Pbounds[2,t] = P[n] - sum(Pbounds[2, :])   # mass differential
#         end

#     end
    
#     # multiplying mass at the bounds with prob of observing the remaining NDtime
#     if RTfit == true
      
#         NDtimedistL = Gamma(γ_shape, γ_scale)
#         NDtimedistR = Gamma(γ_shape1, γ_scale1)

#         tvec = dt .* collect(nT:-1:1)

#         pback = zeros(TT,2)
#         pback[1] = transpose(Pbounds[1,:]) * (pdf.(NDtimedistL, tvec) .* dt)
#         pback[2] = transpose(Pbounds[2,:]) * (pdf.(NDtimedistR, tvec) .* dt)
#         return pback

#     else
#         return P
#     end
# end



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


