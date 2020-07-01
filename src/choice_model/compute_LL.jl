"""
    loglikelihood(θ, data, n)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""

"""
    loglikelihood(θ, data, n)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(θ::θ_expfilter, data, n::Int)

    @unpack h_eta, h_beta = θ.θz
    clickdata = map(data->data.click_data,data)
    choice = map(data->data.choice,data)
    sessbnd = map(data->data.sessbnd,data)

    i_0 = compute_initial_pt(h_eta, h_beta, clickdata, choice, sessbnd)

    return sum(pmap((data, i_0) -> loglikelihood!(θ, data, i_0, n), data, i_0))
    
end


"""
    loglikelihood(θ, data, n)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(θ::θ_expfilter_ce, data, n::Int)

    @unpack h_etaC, h_etaE, h_betaC, h_etaE = θ.θz
    clickdata = map(data->data.click_data,data)
    choice = map(data->data.choice,data)
    sessbnd = map(data->data.sessbnd,data)

    i_0 = compute_initial_pt(h_etaC, h_etaE, h_betaC, h_etaE, clickdata, choice, sessbnd)

    return sum(pmap((data, i_0) -> loglikelihood!(θ, data, i_0, n), data, i_0))
    
end

"""
    (θ::θchoice)(data)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
(θ)(data; n::Int=53) = loglikelihood(θ, data, n)


"""
    loglikelihood!(θ, data, n)
    
Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
function loglikelihood!(θ::DDMθ,data::choicedata,
        i_0::TT, n::Int) where {TT,UU <: Real}

    @unpack θz, lapse, bias = θ
    @unpack B, λ, σ2_a, σ2_i, h_drift_scale = θz
    @unpack click_data, choice = data
    @unpack dt = click_data

    P,M,xc,dx = initialize_latent_model(σ2_i,i_0, B, λ, σ2_a, n, dt, lapse=lapse)

    P = P_single_trial!(θz,P,M,dx,xc,click_data,i_0*h_drift_scale,n)
    log(sum(choice_likelihood!(bias,xc,P,choice,n,dx)))

end


"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)

Given parameters θz progagates P for one trial
"""
function P_single_trial!(θz,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, click_data, scaled_i_0::TT,
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
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,scaled_i_0,M,dx,xc,n,dt)

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


# function loglikelihood(θ::θ_expfilter, data, n::Int)

#     @unpack h_C, h_eta, h_beta = θ.θz
#     clickdata = map(data->data.click_data,data)
#     sessbnd = map(data->data.sessbnd,data)
#     # hack to do constrained optimization with an unconstrained algorithm
#     # the prior is set up so that when dum>0 constraint is not met
#     # a gentle slope is introduced to repush the parameters towards the
#     # allowed space, this constraint arises because exponential filtering
#     # happens in probability space
#     reg = InverseGamma(0.001, 0.1);
#     ep = 0.0001344
#     dum = 1. - h_C - (h_eta*h_beta/(1. - h_beta))

#     if dum < ep
#         dum_intercept = log(pdf.(reg, ep))
#         dum_slope = -(log(pdf.(reg, ep)) - log(pdf.(reg,2*ep)))/ep
#         return dum_slope*(dum-ep) + dum_intercept + length(sessbnd)*log(0.5)
#     else
#         i_0 = compute_initial_pt(h_C,h_eta,h_beta,clickdata, sessbnd)
#         return sum(pmap((data, i_0) -> loglikelihood!(θ, data, i_0, n), data, i_0)) + log(pdf.(reg, dum))
#     end

# end

# """
#     bounded_mass(θ, data, n)
# """
# function bounded_mass(θ::θchoice, data, n::Int)

#     @unpack θz, lapse = θ
#     @unpack σ2_i, B, λ, σ2_a = θz
#     @unpack dt = data[1].click_data

#     P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, lapse=lapse)

#     pmap(data -> bounded_mass!(θ, P, M, dx, xc, data, n), data)

# end


# """
#     bounded_mass!(θ, P, M, dx, xc, data, n)
# """
# function bounded_mass!(θ::θchoice,
#         P::Vector{TT}, M::Array{TT,2}, dx::UU,
#         xc::Vector{TT}, data::choicedata,
#         n::Int) where {TT,UU <: Real}

#     @unpack θz, bias = θ
#     @unpack click_data, choice = data

#     P = P_single_trial!(θz,P,M,dx,xc,click_data,n)
#     choice ? P[n] : P[1]

# end


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
