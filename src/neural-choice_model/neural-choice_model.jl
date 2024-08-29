"""
    choice_neural_optimize(model, options)

Optimize (potentially all) model parameters for a `neural_choiceDDM` using choice and neural data.

Arguments: 

- `model`: an instance of a `neural_choiceDDM`.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function choice_neural_optimize(model::neural_choiceDDM, data;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false)
    
    @unpack θ, n, cross, fit, lb, ub = model
    @unpack f = θ
    
    x0 = PulseInputDDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -joint_loglikelihood(stack(x,c,fit), model, data)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)    
    model.θ = θneural_choice(x, f)

    return model, output

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

"""
function gradient(model::neural_choiceDDM, data)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model, data)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.

"""
function Hessian(model::neural_choiceDDM, data; chunk_size::Int=4)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model, data)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    joint_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function joint_loglikelihood(x::Vector{T}, model::neural_choiceDDM, data) where {T <: Real}

    @unpack θ,n,cross,fit,lb,ub = model
    @unpack f = θ 
    
    model = neural_choiceDDM(θ=θneural_choice(x, f), n=n, cross=cross,fit=fit, lb=lb, ub=ub)
    
    joint_loglikelihood(model, data)

end


"""
    joint_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
joint_loglikelihood(model::neural_choiceDDM, data) = sum(log.(vcat(vcat(joint_likelihood(model, data)...)...)))


"""
    joint_loglikelihood_per_trial(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function joint_loglikelihood_per_trial(model::neural_choiceDDM, data) 
    
    output = joint_likelihood(model, data)
    map(x-> map(x-> sum(log.(x)), x), output)
    
end


"""
    joint_likelihood(model)

Arguments: `neural_choiceDDM` instance

Returns: `array` of `array` of P(d, Y|θ)
"""
function joint_likelihood(model::neural_choiceDDM, data)
    
    @unpack θ,n,cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> 
            joint_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)
    
end


"""
"""
function joint_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack choice = data
    @unpack θz, bias, lapse = θ
    
    c, P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)
    
    return vcat(c, sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2)
     
end


"""
"""
function posterior(model::neural_choiceDDM, data)
    
    @unpack θ,n,cross = model
    @unpack θy,θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> posterior(θ, θy, data, P, M, xc, dx, n, cross), data), data, θy)

end


"""
"""
function posterior(θ::θneural_choice, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack θz, bias, lapse = θ
    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data, choice = data
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
        
        (t == length(time_bin)) && (P = choice_likelihood!(bias,xc,P,choice,n,dx))
        
        c[t] = sum(P)
        P /= c[t]
        α[:,t] = P

    end   

    P = ones(Float64,n) #initialze backward pass with all 1's
    β[:,end] = P

    @inbounds for t = length(time_bin)-1:-1:1

        (t+1 == length(time_bin)) && (P = choice_likelihood!(bias,xc,P,choice,n,dx))            
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
function forward(model::neural_choiceDDM, data)
    
    @unpack θ,n,cross = model
    @unpack θy,θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> forward(θ, θy, data, P, M, xc, dx, n, cross), data), data, θy)

end


"""
"""
function forward(θ::θneural_choice, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack θz, bias, lapse = θ
    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data, choice = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks
    
    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)
    
    time_bin = (-(pad-1):nT+pad) .- delay

    c = Vector{T1}(undef, length(time_bin))
    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks   
    α = Array{Float64,2}(undef, n, length(time_bin))
        
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

    return α, xc

end