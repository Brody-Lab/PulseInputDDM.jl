"""
"""
@with_kw mutable struct choiceoptions
    fit::Vector{Bool} = vcat(trues(dimz+2))
    lb::Vector{Float64} = vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005], [-30, 0.])
    ub::Vector{Float64} = vcat([Inf, Inf, 10., Inf, Inf, 1.2,  1.], [30, 1.])
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01])
end


"""
    θchoice{T1, T<:Real} <: DDMθ

Fields:

- `θz` is a type that contains the parameters related to the latent variable model.
- `bias` is the choice bias parameter.
- `lapse` is the lapse parameter.

Example:

```julia
θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05), bias=1., lapse=0.05)
```
"""
@with_kw struct θchoice{T1, T<:Real} <: DDMθ
    θz::T1 = θz()
    bias::T = 1.
    lapse::T = 0.05
end


"""
    choicedata{T1} <: DDMdata

Fields:

- `click_data` is a type that contains all of the parameters related to click input.
- `choice` is the choice data for a single trial.

Example:

```julia
ntrials, dt, centered = 1, 1e-2, false
θ = θchoice()
clicks, choices = rand(θ, ntrials)
binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)
inputs = choiceinputs(clicks, binned_clicks, dt, centered)
choicedata(inputs, choices)
```
"""
@with_kw struct choicedata{T1} <: DDMdata
    click_data::T1
    choice::Bool
end


"""
    choiceDDM{T,U} <: DDM

Fields:

- `θ` is a type that contains all of the model parameters.
- `data` is a type that contains all of the data (inputs and choices).

Example:

```julia
ntrials, dt, centered = 1, 1e-2, false
θ = θchoice()
clicks, choices = rand(θ, ntrials)
binned_clicks = bin_clicks(clicks, centered=centered, dt=dt)
inputs = choiceinputs(clicks, binned_clicks, dt, centered)
data = choicedata(inputs, choices)
choiceDDM(θ, data)
```
"""
@with_kw struct choiceDDM{T,U} <: DDM
    θ::T = θchoice()
    data::U
end


"""
    optimize_model(data, options)

Optimize model parameters. `data` is a type that contains the click data and the choices.
options is a type that contains the initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize(data, options::choiceoptions; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false,
        time_limit::Float64=170000., show_every::Int=10, σ::Vector{Float64}=eps()*ones(dimz), 
        μ::Vector{Float64}=eps()*ones(dimz), do_prior::Bool=false, cross::Bool=false)

    @unpack fit, lb, ub, x0 = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data; n=n, cross=cross))
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled, time_limit=time_limit, show_every=show_every)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model, output

end


"""
    loglikelihood(x, data)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood(x::Vector{T1}, data; n::Int=53, cross::Bool=false) where {T1 <: Real}

    θ = Flatten.reconstruct(θchoice(), x)
    loglikelihood(θ, data; n=n, cross=cross)

end


"""
    gradient(model)

Given a DDM model (parameters and data), compute the gradient.
"""
function gradient(model::choiceDDM; n::Int=53, cross::Bool=false)

    @unpack θ, data = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data; n=n, cross=cross)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model)

Given a DDM model (parameters and data), compute the Hessian.
"""
function Hessian(model::T; n::Int=53, cross::Bool=false) where T <: DDM

    @unpack θ, data = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data; n=n, cross=cross)

    ForwardDiff.hessian(ℓℓ, x)

end



"""
"""
θ2(θ) = θchoice(θz=θz(σ2_i = θ.θz.σ2_i^2, B = θ.θz.B, λ = θ.θz.λ, 
        σ2_a = θ.θz.σ2_a^2, σ2_s = θ.θz.σ2_s^2, 
        ϕ = θ.θz.ϕ, τ_ϕ = θ.θz.τ_ϕ), bias=θ.bias, lapse=θ.lapse)   


"""
"""
θexp(θ) = θchoice(θz=θz(σ2_i = exp(θ.θz.σ2_i), B = θ.θz.B, λ = θ.θz.λ, 
        σ2_a = exp(θ.θz.σ2_a), σ2_s = exp(θ.θz.σ2_s), 
        ϕ = θ.θz.ϕ, τ_ϕ = θ.θz.τ_ϕ), bias=θ.bias, lapse=θ.lapse)   
    
    
"""
    loglikelihood(θ, data)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(θ::θchoice, data; n::Int=53, cross::Bool=false)
    
    θ = θ2(θ)

    @unpack θz, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, lapse=lapse)
    sum(pmap(data -> loglikelihood!(θ, P, M, dx, xc, data, n=n, cross=cross), data))

end


"""
    (θ::θchoice)(data)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
(θ::θchoice)(data; n::Int=53, cross::Bool=false) = loglikelihood(θ, data; n=n, cross=cross)


"""
    loglikelihood!(θ, P, M, dx, xc, data, n)

Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
function loglikelihood!(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data::choicedata;
        n::Int=53, cross::Bool=false) where {TT,UU <: Real}

    @unpack θz, bias = θ
    @unpack click_data, choice = data

    P = P_single_trial!(θz,P,M,dx,xc,click_data,n; cross=cross)
    log(sum(choice_likelihood!(bias,xc,P,choice,n,dx)))

end


"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)

Given parameters θz progagates P for one trial
"""
function P_single_trial!(θz,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, click_data,
        n::Int; cross::Bool=false,
        keepP::Bool=false) where {TT,UU <: Real}

    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    #empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)
    
    if keepP
        PS = Vector{Vector{Float64}}(undef, nT)
    end

    @inbounds for t = 1:nT

        #maybe only pass one L,R,nT?
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        if keepP
            PS[t] = P
        end


    end

    if keepP
        return PS
    else
        return P
    end

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


"""
    bounded_mass(θ, data, n)
"""
function bounded_mass(θ::θchoice, data, n::Int; cross::Bool=false)

    @unpack θz, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, lapse=lapse)

    pmap(data -> bounded_mass!(θ, P, M, dx, xc, data, n; cross=cross), data)

end


"""
    bounded_mass!(θ, P, M, dx, xc, data, n)
"""
function bounded_mass!(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data::choicedata,
        n::Int; cross::Bool=false) where {TT,UU <: Real}

    @unpack θz, bias = θ
    @unpack click_data, choice = data

    P = P_single_trial!(θz,P,M,dx,xc,click_data,n; cross=cross)
    choice ? P[n] : P[1]

end


#=
Backward pass, for one day when I might need to compute the posterior again.

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