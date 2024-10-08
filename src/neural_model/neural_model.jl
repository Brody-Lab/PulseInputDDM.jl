"""
    flatten(θ)

Extract parameters `neuralDDM` and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θneural)

    @unpack θy, θz = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `neuralDDM`.
"""
function gradient(model::neuralDDM, data)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model, data)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `neuralDDM`.

Arguments:

- `model`: instance of `neuralDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.

"""
function Hessian(model::neuralDDM, data; chunk_size::Int=4)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model, data)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
Optimize model parameters for a `neuralDDM`.

 $(SIGNATURES)

Arguments: 

- `model`: an instance of a `neuralDDM`.
- `data`: an instance of a `neuralDDM`.

Returns:

- `model`: an instance of a `neuralDDM`.

"""
function fit(model::neuralDDM, data;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false)
    
    @unpack fit, lb, ub, θ, n, cross = model
    @unpack f = θ
    
    x0 = PulseInputDDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model, data)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model.θ = θneural(x, f)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `neuralDDM`

"""
function loglikelihood(x::Vector{T}, model::neuralDDM, data) where {T <: Real}
    
    @unpack θ,n,cross,fit,lb,ub = model
    @unpack f = θ 

    model = neuralDDM(θ=θneural(x, f), n=n, cross=cross,fit=fit, lb=lb, ub=ub)

    loglikelihood(model, data)

end


"""
    loglikelihood(model)

Arguments: `neuralDDM` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood(model::neuralDDM, data)
    
    sum(sum.(loglikelihood_pertrial(model, data)))

end


"""
    loglikelihood_pertrial(model)

Arguments: `neuralDDM` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood_pertrial(model::neuralDDM, data)
    
    @unpack θ,n,cross = model
    @unpack θz, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> loglikelihood(θz,θy,data, P, M, xc, dx, n, cross), data), data, θy)

end


"""
"""
loglikelihood(θz,θy,data::neuraldata, P::Vector{T1}, M::Array{T1,2},
    xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real} = sum(log.(likelihood(θz,θy,data,P,M,xc,dx,n,cross)[1]))


#=
function likelihood(θz,θy,data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    alpha = log.(P)

    @inbounds for t = 1:length(time_bin)
        
        mm = maximum(alpha)
        py = vcat(map(xc-> sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt), k[t]), spikes, θy, λ0)), xc)...)

        if time_bin[t] >= 1
            
            any(t .== nL) ? sL = sum(La[t .== nL]) : sL = zero(T1)
            any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = zero(T1)
            σ2 = σ2_s * (sL + sR);   μ = -sL + sR

            if (sL + sR) > zero(T1)
                transition_M!(F,σ2+σ2_a*dt,λ, μ, dx, xc, n, dt)
                alpha = log.((exp.(alpha .- mm)' * F)') .+ mm .+ py
            else
                alpha = log.((exp.(alpha .- mm)' * M)') .+ mm .+ py
            end
            
        else
            alpha = alpha .+ py
        end
                       
    end

    return exp(logsumexp(alpha)), exp.(alpha)

end
=#

"""
"""
function likelihood(θz,θy,data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    c = Vector{T1}(undef, length(time_bin))

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end

        #weird that this wasn't working....
        #P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
        #                        k[t]), spikes, θy, λ0))), xc)...)
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]

    end

    return c, P

end


"""
"""
function posterior(model::neuralDDM, data)
    
    @unpack θ,n,cross = model
    @unpack θz, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> posterior(θz,θy,data, P, M, xc, dx, n, cross), data), data, θy)

end


"""
"""
function posterior(θz::θz, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks
    
    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)
    
    time_bin = (-(pad-1):nT+pad) .- delay

    c = Vector{T1}(undef, length(time_bin))
    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks   
    α = Array{Float64,2}(undef, n, length(time_bin))
    β = Array{Float64,2}(undef, n, length(time_bin))
        
    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]
        α[:,t] = P

    end   

    P = ones(Float64,n) #initialze backward pass with all 1's
    β[:,end] = P

    @inbounds for t = length(time_bin)-1:-1:1

        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t+1]) * dt),
                k[t+1]), spikes, θy, λ0))), xc)...))
            
        if time_bin[t] >= 0
            P,F = backward_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t+1], nL, nR, La, Ra, M, dx, xc, n, dt)
        end
        
        P /= c[t+1]
        β[:,t] = P

    end

    return α, β, xc

end


"""
"""
function logistic!(x::T) where {T <: Any}

    if x >= 0.
        x = exp(-x)
        x = 1. / (1. + x)
    else
        x = exp(x)
        x = x / (1. + x)
    end

    return x

end