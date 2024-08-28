"""
    fit(model, options)

fit model parameters for a `choiceDDM`.

Returns:

- `model`: an instance of a `choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

Arguments:

- `model`: an instance of a `choiceDDM`.

"""
function fit(model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, 
        Vector{choicedata{choiceinputs{clicks, binned_clicks}}}};
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false, time_limit::Float64=170000., show_every::Int=10)

    @unpack fit, lb, ub, θ, n, cross = model
    
    x0 = collect(Flatten.flatten(θ))

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model, data)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled, time_limit=time_limit, show_every=show_every)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ=θ, n=n, cross=cross, fit=fit, lb=lb, ub=ub)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood(x::Vector{T1}, model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, Vector{choicedata{choiceinputs{clicks, binned_clicks}}}}) where {T1 <: Real}

    @unpack n, cross = model
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ=θ, n=n, cross=cross)
    loglikelihood(model, data)

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `choiceDDM`.
"""
function gradient(model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, Vector{choicedata{choiceinputs{clicks, binned_clicks}}}})

    @unpack θ = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, model, data)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `choiceDDM`.
"""
function Hessian(model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, Vector{choicedata{choiceinputs{clicks, binned_clicks}}}})

    @unpack θ = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, model, data)

    ForwardDiff.hessian(ℓℓ, x)

end

    
"""
    loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, Vector{choicedata{choiceinputs{clicks, binned_clicks}}}})
    
    @unpack θ, n, cross = model
    @unpack θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    sum(pmap(data -> loglikelihood!(θ, P, M, dx, xc, data, n, cross), data))

end


"""
    loglikelihood!(θ, P, M, dx, xc, data, n, cross)

Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
loglikelihood!(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data::choicedata,
        n::Int, cross::Bool) where {TT,UU <: Real} = log(likelihood!(θ, P, M, dx, xc, data, n, cross))


"""
    likelihood(model)

Given parameters θ and data (inputs and choices) computes the likehood of the choice for all trials
"""
function likelihood(model::choiceDDM, data::Vector{choicedata{choiceinputs{clicks, binned_clicks}}})
    
    @unpack θ, n, cross = model
    @unpack θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    pmap(data -> likelihood!(θ, P, M, dx, xc, data, n, cross), data)

end


"""
    likelihood!(θ, P, M, dx, xc, data, n, cross)

Given parameters θ and data (inputs and choices) computes the LL for one trial
"""
function likelihood!(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data::choicedata,
        n::Int, cross::Bool) where {TT,UU <: Real}

    @unpack θz, bias, lapse = θ
    @unpack click_data, choice = data

    P = P_single_trial!(θz,P,M,dx,xc,click_data,n,cross)
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2

end


"""
    P_single_trial!(θz, P, M, dx, xc, click_data, n)

Given parameters θz progagates P for one trial
"""
function P_single_trial!(θz,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, click_data,
        n::Int, cross::Bool;
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
function choice_likelihood!(bias::TT, xc::Vector{TT}, P::Vector{VV},
                 pokedR::Bool, n::Int, dx::UU) where {TT,UU,VV <: Any}

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
    posterior(model)

"""
function posterior(model::choiceDDM, data::Union{choicedata{choiceinputs{clicks, binned_clicks}}, Vector{choicedata{choiceinputs{clicks, binned_clicks}}}})
    
    @unpack θ, n, cross = model
    @unpack θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    pmap(data -> posterior(θ, data, P, M, dx, xc, data, n, cross), data)

end


"""
    posterior(θz, P, M, dx, xc, click_data, n)

"""
function posterior(θ::θchoice, data::choicedata,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, click_data,
        n::Int, cross::Bool) where {TT,UU <: Real}

    @unpack θz, bias, lapse = θ
    @unpack click_data, choice = data
    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    F = zeros(TT,n,n)
    α = Array{Float64,2}(undef, n, nT)
    β = Array{Float64,2}(undef, n, nT)
    c = Vector{TT}(undef, nT)

    @inbounds for t = 1:nT

        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        
        (t == nT) && (P = choice_likelihood!(bias,xc,P,choice,n,dx))
        
        c[t] = sum(P)
        P /= c[t]
        α[:,t] = P

    end
    
    P = ones(Float64,n) #initialze backward pass with all 1's
    β[:,end] = P

    @inbounds for t = nT-1:-1:1

        (t+1 == nT) && (P = choice_likelihood!(bias,xc,P,choice,n,dx))            
        P,F = backward_one_step!(P, F, λ, σ2_a, σ2_s, t+1, nL, nR, La, Ra, M, dx, xc, n, dt)        
        P /= c[t+1]
        β[:,t] = P

    end

    return α, β, xc
   
end


"""
    forward(model)

"""
function forward(model::choiceDDM, data::Vector{choicedata{choiceinputs{clicks, binned_clicks}}})
    
    @unpack θ, n, cross = model
    @unpack θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    pmap(data -> forward(θ, P, M, dx, xc, data, n, cross), data)

end


"""
    forward(θz, P, M, dx, xc, click_data, n)

"""
function forward(θ::θchoice,
        P::Vector{TT}, M::Array{TT,2}, dx::UU,
        xc::Vector{TT}, data,
        n::Int, cross::Bool) where {TT,UU <: Real}

    @unpack θz, bias, lapse = θ
    @unpack click_data, choice = data
    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θz
    @unpack binned_clicks, clicks, dt = click_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    F = zeros(TT,n,n)
    α = Array{Float64,2}(undef, n, nT)
    c = Vector{TT}(undef, nT)

    @inbounds for t = 1:nT

        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
                
        c[t] = sum(P)
        P /= c[t]
        α[:,t] = P

    end

    return α, xc
   
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