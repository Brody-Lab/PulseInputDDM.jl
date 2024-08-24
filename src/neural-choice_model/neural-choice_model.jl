"""
"""
@with_kw struct neural_choice_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function neural_choice_options(f; remap::Bool=false)
    
    nparams, ncells = nθparams(f)
    fit = vcat(trues(dimz+2), trues.(nparams)...)
        
    lb = Vector(undef, sum(ncells))
    ub = Vector(undef, sum(ncells))
    
    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb[i] = [-10]
            ub[i] = [10]
        elseif vcat(f...)[i] == "Sigmoid"
            lb[i] = [-100.,0.,-10.,-10.]
            ub[i] = [100.,100.,10.,10.]
        elseif vcat(f...)[i] == "Softplus_negbin"
            lb[i] = [0, -10]
            ub[i] = [Inf, 10]
        end
    end
    
    if remap
        lb = vcat([-10., 8.,  -5., -20.,   -3.,   1e-3, 0.005], [-10, 0.], vcat(lb...))
        ub = vcat([ 10., 40., 5.,  20.,    3.,   1.2,  1.],    [10, 1.],  vcat(ub...));
    else
        lb = vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.], vcat(lb...))
        ub = vcat([100., 40., 5., 400., 10., 1.2,  1.], [10, 1.], vcat(ub...));
    end

    neural_choice_options(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θneural_choice(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(dimz + 2,dimz + 2 .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_choice(θz(x[1:dimz]...), x[dimz+1], x[dimz+2], θy, f)

end


"""
    flatten(θ)

Extract parameters related to a `neural_choiceDDM` from an instance of `θneural_choice` and returns an ordered vector.
```
"""
function flatten(θ::θneural_choice)

    @unpack θy, θz, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    all_Softplus(data)

Returns: `array` of `array` of `string`, of all Softplus
"""
function all_Softplus(data)
    
    ncells = getfield.(first.(data), :ncells)
    f = repeat(["Softplus"], sum(ncells))
    borg = vcat(0,cumsum(ncells))
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
end


"""
    choice_neural_optimize(model, options)

Optimize (potentially all) model parameters for a `neural_choiceDDM` using choice and neural data.

Arguments: 

- `model`: an instance of a `neural_choiceDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function choice_neural_optimize(model::neural_choiceDDM, data, options::neural_choice_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, n, cross = model
    @unpack f = θ
    
    x0 = PulseInputDDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -joint_loglikelihood(stack(x,c,fit), model, data; remap=remap)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    
    model = neural_choiceDDM(θneural_choice(x, f), n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
"""
θz2(θ::θz) = θz(σ2_i=θ.σ2_i^2, B=θ.B, λ=θ.λ, 
        σ2_a=θ.σ2_a^2, σ2_s=θ.σ2_s^2, ϕ=θ.ϕ, τ_ϕ=θ.τ_ϕ)


"""
"""
θzexp(θ::θz) = θz(σ2_i=exp(θ.σ2_i), B=θ.B, λ=θ.λ, 
        σ2_a=exp(θ.σ2_a), σ2_s=exp(θ.σ2_s), ϕ=exp(θ.ϕ), τ_ϕ=θ.τ_ϕ)


"""
"""
invθzexp(θ::θz) = θz(σ2_i=log(θ.σ2_i), B=θ.B, λ=θ.λ, 
        σ2_a=log(θ.σ2_a), σ2_s=log(θ.σ2_s), ϕ=log(θ.ϕ), τ_ϕ=θ.τ_ϕ)


"""
"""
θ2(θ::θneural_choice) = θneural_choice(θz=θz2(θ.θz), bias=θ.bias, lapse=logistic(θ.lapse), θy=θ.θy, f=θ.f)


"""
"""
θexp(θ::θneural_choice) = θneural_choice(θz=θzexp(θ.θz), bias=θ.bias, lapse=logistic(θ.lapse), θy=θ.θy, f=θ.f)


"""
"""
invθz2(θ::θz) = θz(σ2_i=abs(sqrt(θ.σ2_i)), B=θ.B, λ=θ.λ, 
        σ2_a=abs(sqrt(θ.σ2_a)), σ2_s=abs(sqrt(θ.σ2_s)), ϕ=θ.ϕ, τ_ϕ=θ.τ_ϕ)


"""
"""
invθ2(θ::θneural_choice) = θneural_choice(θz=invθz2(θ.θz), bias=θ.bias, lapse=logit(θ.lapse), θy=θ.θy, f=θ.f)


"""
"""
invθexp(θ::θneural_choice) = θneural_choice(θz=invθzexp(θ.θz), bias=θ.bias, lapse=logit(θ.lapse), θy=θ.θy, f=θ.f)


"""
    P_goright(model)

Given an instance of `choiceDDM` computes the probabilty of going right for each trial.
"""
function P_goright(model::neural_choiceDDM, data)
    
    @unpack θ, n, cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data
       
    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)
    
    map((data, θy) -> pmap(data -> 
            P_goright(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)

end


"""
"""
function P_goright(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack choice = data
    @unpack θz, bias, lapse = θ
    
    P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)[2]
    sum(choice_likelihood!(bias,xc,P,true,n,dx)) * (1 - lapse) + lapse/2
    
end


"""
    gradient(model; remap)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `remap`: For considering parameters in variance of std space.

"""
function gradient(model::neural_choiceDDM, data; remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model, data; remap=remap)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
- `remap`: For considering parameters in variance of std space.

"""
function Hessian(model::neural_choiceDDM, data; chunk_size::Int=4, remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model, data; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    joint_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function joint_loglikelihood(x::Vector{T}, model::neural_choiceDDM, data; remap::Bool=false) where {T <: Real}
    
    @unpack θ,n,cross = model
    @unpack f = θ 
    
    if remap
        model = neural_choiceDDM(θexp(θneural_choice(x, f)), n, cross)
    else
        model = neural_choiceDDM(θneural_choice(x, f), n, cross)
    end
    
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