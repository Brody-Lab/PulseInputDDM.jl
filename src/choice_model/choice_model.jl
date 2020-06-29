"""
    optimize_model(data, options, n; x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)

Optimize model parameters. data is a type that contains the click data and the choices.
options is a type that contains the initial values, boundaries,
and specification of which parameters to fit.

BACK IN THE DAY TOLS WERE: x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

"""
function optimize(data, modeltype, options::DDMθoptions, n::Int;
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-10,
        iterations::Int=Int(5e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false)

    @unpack fit, lb, ub, x0 = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = objectivefn(modeltype, x,c, fit, data, n)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = reconstruct_model(x, modeltype)
    model = choiceDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model, output

end

"""
"""
function objectivefn(objfname, x, c, fit, data, n)
    if objfname == "expfilter"
        o = -loglikelihood_expfilter(stack(x,c,fit), data, n)
    elseif objfname == "expfilter_ce"
        o = -loglikelihood_expfilter_ce(stack(x,c,fit), data, n)
    else
        error("Unknown modeltype $objfname")
    end
end

"""
    loglikelihood_expfilter(x, data, n)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood_expfilter(x::Vector{T1}, data, n::Int) where {T1 <: Real}

    θ = Flatten.reconstruct(θ_expfilter(), x)
    loglikelihood(θ, data, n)

end

"""
    loglikelihood_expfilter_ce(x, data, n)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood_expfilter_ce(x::Vector{T1}, data, n::Int) where {T1 <: Real}

    θ = Flatten.reconstruct(θ_expfilter_ce(),x)
    loglikelihood(θ, data, n)

end
"""
"""

function reconstruct_model(x::Vector{T1}, modeltype) where {T1 <: Real}
    if modeltype == "expfilter"
        θ = Flatten.reconstruct(θ_expfilter(), x)
    elseif modeltype == "expfilter_ce"
        θ = Flatten.reconstruct(θ_expfilter_ce(), x)
    end
    return θ
end

"""
    gradient(model, n)

Given a DDM model (parameters and data), compute the gradient.
# """
function gradient(model::T, n::Int) where T <: DDM

    @unpack θ, data = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, n)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    Hessian(model, n)

Given a DDM model (parameters and data), compute the Hessian.
"""
function Hessian(model::T, n::Int) where T <: DDM

    @unpack θ, data = model
    x = [Flatten.flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, n)

    ForwardDiff.hessian(ℓℓ, x)

end
