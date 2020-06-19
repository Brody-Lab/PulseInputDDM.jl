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
    bias::T = 0.
    lapse::T = 0.01
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
    optimize_model(data, options, n; x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)

Optimize model parameters. data is a type that contains the click data and the choices.
options is a type that contains the initial values, boundaries,
and specification of which parameters to fit.

BACK IN THE DAY TOLS WERE: x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

"""
function optimize(data, options::choiceoptions, n::Int;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false)

    @unpack fit, lb, ub, x0 = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, n)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model, output

end


"""
    loglikelihood(x, data, n)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood(x::Vector{T1}, data, n::Int) where {T1 <: Real}

    θ = Flatten.reconstruct(θchoice(), x)
    loglikelihood(θ, data, n)

end


"""
    gradient(model, n)

Given a DDM model (parameters and data), compute the gradient.
"""
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
