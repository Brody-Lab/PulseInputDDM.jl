"""
    choice_optimize(data)

Optimize model parameters for a `neural_choiceDDM` using choice data. Neural tuning parameters ([`θy`](@ref)) are initialized by fitting a the noiseless DDM model first ([`noiseless_neuralDDM`](@ref)).

Arguments:

- `data`: the output of [`load_neural_data`](@ref) with the format as described in its docstring.

Returns

- `model`: a module-defined type that organizes the `data` and parameters from the fit (as well as a few other things that are necessary for re-computing things the way they were computed here (e.g. `n`)
- `options`: some details related to the optimzation, such as which parameters were fit, and the upper and lower bounds of those parameters.

"""
function choice_optimize(data;
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
    model, = choice_optimize(model, options; show_trace=show_trace, f_tol=f_tol, 
        iterations=iterations, outer_iterations=outer_iterations, remap=remap)

    return model, options

end


"""
    choice_optimize(model, options)

Optimize choice-related model parameters for a `neural_choiceDDM` using choice data.

Arguments: 

- `model`: an instance of a `neural_choiceDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function choice_optimize(model::neural_choiceDDM, options::neural_choice_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
    @unpack f = θ
    
    x0 = PulseInputDDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -choice_loglikelihood(stack(x,c,fit), model; remap=remap)
    
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
    choice_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function choice_loglikelihood(x::Vector{T}, model::neural_choiceDDM; remap::Bool=false) where {T <: Real}
    
    @unpack data,θ,n,cross = model
    @unpack f = θ 
    if remap
        model = neural_choiceDDM(θ2(θneural_choice(x, f)), data, n, cross)
    else
        model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    end
    choice_loglikelihood(model)

end


"""
    choice_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
choice_loglikelihood(model::neural_choiceDDM) = sum(log.(vcat(choice_likelihood(model)...)))


"""
"""
function choice_loglikelihood_per_trial(model::neural_choiceDDM) 
    
    output = choice_likelihood(model)
    map(x-> map(x-> sum(log.(x)), x), output)
    
end


"""
    choice_likelihood(model)

Arguments: `neural_choiceDDM` instance

Returns: `array` of `array` of P(d|θ, Y)
"""
function choice_likelihood(model::neural_choiceDDM)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data -> 
            choice_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)
    
end


"""
"""
function choice_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}
    
    @unpack choice = data
    @unpack θz, bias, lapse = θ
    
    P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)[2]
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2
    
end