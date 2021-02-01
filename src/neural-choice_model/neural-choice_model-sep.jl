"""
    sep_choice_neural_optimize(data)

Optimize model parameters for a `neural_choiceDDM`. Neural tuning parameters ([`θy`](@ref)) are initialized by fitting a the noiseless DDM model first ([`noiseless_neuralDDM`](@ref)).

Arguments:

- `data`: the output of [`load_neural_data`](@ref) with the format as described in its docstring.

Returns

- `model`: a module-defined type that organizes the `data` and parameters from the fit (as well as a few other things that are necessary for re-computing things the way they were computed here (e.g. `n`)
- `options`: some details related to the optimzation, such as which parameters were fit, and the upper and lower bounds of those parameters.

"""
function sep_choice_neural_optimize(data;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        show_trace::Bool=true,  extended_trace::Bool=false, scaled::Bool=false,
        outer_iterations::Int=Int(1e1), iterations::Int=Int(2e3),
        n::Int=53, cross::Bool=false,
        x0_z::Vector{Float64}=[0.1, 15., -0.1, 20., 0.8, 0.01, 0.008],
        f::Vector{Vector{String}}=all_Softplus(data),
        remap::Bool=false) 
               
    x0 = vcat(x0_z, [0., 1e-1], collect.(Flatten.flatten.(vcat(θy0(data,f)...)))...)
    nparams, = nθparams(f)    
    fit = vcat(trues(dimz), trues(2), trues.(nparams)...)
    options = neural_choice_options(f)
    options = neural_choice_options(fit=fit, lb=options.lb, ub=options.ub)
        
    model = neural_choiceDDM(θneural_choice(x0, f), data, n, cross)  
    model, = sep_choice_neural_optimize(model, options; show_trace=show_trace, f_tol=f_tol, 
        iterations=iterations, outer_iterations=outer_iterations, remap=remap)

    return model, options

end


"""
    sep_choice_neural_optimize(model, options)

Optimize (potentially all) model parameters for a `neural_choiceDDM` using choice and neural data.

Arguments: 

- `model`: an instance of a `neural_choiceDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function sep_choice_neural_optimize(model::neural_choiceDDM, options::neural_choice_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
    @unpack f = θ
    
    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -sep_joint_loglikelihood(stack(x,c,fit), model; remap=remap)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    
    model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
    sep_gradient(model; remap)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `remap`: For considering parameters in variance of std space.

"""
function sep_gradient(model::neural_choiceDDM; remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -sep_joint_loglikelihood(x, model; remap=remap)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    sep_Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
- `remap`: For considering parameters in variance of std space.

"""
function sep_Hessian(model::neural_choiceDDM; chunk_size::Int=4, remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -sep_joint_loglikelihood(x, model; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    sep_joint_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function sep_joint_loglikelihood(x::Vector{T}, model::neural_choiceDDM; remap::Bool=false) where {T <: Real}
    
    @unpack data,θ,n,cross = model
    @unpack f = θ 
    
    if remap
        model = neural_choiceDDM(θexp(θneural_choice(x, f)), data, n, cross)
    else
        model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    end
    
    sep_joint_loglikelihood(model)

end


"""
    sep_joint_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
sep_joint_loglikelihood(model::neural_choiceDDM) = sum(log.(vcat(vcat(sep_joint_likelihood(model)...)...)))


"""
    sep_joint_loglikelihood_per_trial(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function sep_joint_loglikelihood_per_trial(model::neural_choiceDDM) 
    
    output = sep_joint_likelihood(model)
    map(x-> map(x-> sum(log.(x)), x), output)
    
end


"""
    sep_joint_likelihood(model)

Arguments: `neural_choiceDDM` instance

Returns: `array` of `array` of P(d, Y|θ)
"""
function sep_joint_likelihood(model::neural_choiceDDM)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> 
            sep_joint_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)
    
end


"""
"""
function sep_joint_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack choice = data
    @unpack θz, bias, lapse = θ
    @unpack spikes, input_data = data
    @unpack λ0 = input_data
    
    #c, P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)
    output = map((θy, spikes, λ0) -> likelihood(θz, [θy], [spikes], input_data, [λ0], P, M, xc, dx, n, cross), θy, spikes, λ0)
    c = getindex.(output, 1)
    P = getindex.(output, 2)
    
    choicepart = exp(mean(log.(map(P-> sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2, P))))
    
    return vcat(c..., choicepart)
     
end


"""
"""
function likelihood(θz,θy, spikes, input_data, λ0,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack binned_clicks, clicks, dt, centered, delay, pad = input_data
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
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]

    end

    return c, P

end

#=

"""
"""
function loglikelihood(py, θ)
    
    @unpack m, K = θ
    
    T = size(py,1)
    ps = Vector(undef, size(py,1))
    p = 1/K * ones(K)

    alpha = log.(p)
    
    @inbounds for t in 1:T
        mm = maximum(alpha)
        alpha = log.((exp.(alpha .- mm)' * m)') .+ mm .+ py[t,:]
        ps[t] = exp.(alpha)
    end
    
    return logsumexp(alpha), ps
        
end 

=#