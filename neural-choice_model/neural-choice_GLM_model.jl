"""
"""
@with_kw struct neural_choice_GLM_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function neural_choice_GLM_options(f)
    
    nparams, ncells = nθparams(f)
    fit = vcat(trues(1+2), trues.(nparams)...)
        
    lb = Vector(undef, sum(ncells))
    ub = Vector(undef, sum(ncells))
    
    for i in 1:sum(ncells)
        lb[i] = [-10]
        ub[i] = [10]
    end
    lb = vcat([-5.], [-30, 0.], vcat(lb...))
    ub = vcat([ 5.], [30, 1.], vcat(ub...));

    neural_choice_GLM_options(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θneural_choice_GLM(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(1 + 2, 1 + 2 .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_choice_GLM(x[1], x[2], x[3], θy, f)

end


"""
    flatten(θ)

Extract parameters related to a `neural_choice_GLM_DDM` from an instance of `θneural_choice_GLM` and returns an ordered vector.
```
"""
function flatten(θ::θneural_choice_GLM)

    @unpack θy, stim, bias, lapse = θ
    vcat(stim, bias, lapse,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    optimize(model, options)

Optimize model parameters for a `neural_choice_GLM_DDM`.

Arguments: 

- `model`: an instance of a `neural_choice_GLM_DDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choice_GLM_DDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::neural_choice_GLM_DDM, options::neural_choice_GLM_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
    @unpack f = θ
    
    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    
    model = neural_choice_GLM_DDM(θneural_choice_GLM(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, model::neural_choice_GLM_DDM) where {T <: Real}
    
    @unpack data,θ,n,cross = model
    @unpack f = θ 
    model = neural_choice_GLM_DDM(θneural_choice_GLM(x, f), data, n, cross)
    loglikelihood(model)

end


"""
    loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
loglikelihood(model::neural_choice_GLM_DDM) = sum(max.(log.(vcat(likelihood(model)...)), log(eps())))


"""
    likelihood(model)

Arguments: `neural_choice_GLM_DDM` instance

Returns: `array` of `array` of P(d|θ, ΣY, ΔLR_T)
"""
function likelihood(model::neural_choice_GLM_DDM)
    
    @unpack data,θ,n,cross = model
    @unpack θy = θ

    map((data, θy) -> likelihood.(Ref(θ),Ref(θy),data), data, θy)
    
end


"""
"""
function likelihood(θ, θy, data::neuraldata)
    
    @unpack choice, spikes, input_data = data
    @unpack pad = input_data
    @unpack stim, bias, lapse = θ
    
    Σspikes = sum.(spikes)       
    ΔLR = diffLR(data)
    ΔLR = ΔLR[length(ΔLR)-pad]  
    pdf(Bernoulli((lapse/2) + (1. - lapse) * logistic(stim*ΔLR + sum(getfield.(θy, :c) .* Σspikes) + bias)), choice)

end