"""
    choiceDDM_dx(θ, data, dx, cross)

Fields:

- `θ`: a instance of the module-defined class `θchoice` that contains all of the model parameters for a `choiceDDM`
- `data`: an `array` where each entry is the module-defined class `choicedata`, which contains all of the data (inputs and choices).
- `dx`: width of spatial bin (defaults to 0.25).
- `cross`: whether or not to use cross click adaptation (defaults to false).
"""
@with_kw struct choiceDDM_dx{T,U,V} <: DDM
    θ::T = θchoice()
    data::U
    dx::Float64=0.25
    cross::Bool=false
    θprior::V = θprior()
end



"""
    optimize(model, options)

Optimize model parameters for a `choiceDDM`.

Returns:

- `model`: an instance of a `choiceDDM_dx`.
- `output`: results from [`Optim.optimize`](@ref).

Arguments:

- `model`: an instance of a `choiceDDM_dx`.
- `options`: module-defind type that contains the upper (`ub`) and lower (`lb`) boundaries and specification of which parameters to fit (`fit`).

"""
function optimize(model::choiceDDM_dx, options::choiceoptions; 
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false, time_limit::Float64=170000., show_every::Int=10)

    @unpack fit, lb, ub = options
    @unpack θ, data, dx, cross, θprior = model
    
    x0 = collect(Flatten.flatten(θ))

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), model) + logprior(stack(x,c,fit), θprior))
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled, time_limit=time_limit, show_every=show_every)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM_dx(θ, data, dx, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Given a vector of parameters and a type containing the data related to the choice DDM model, compute the LL.

See also: [`loglikelihood`](@ref)
"""
function loglikelihood(x::Vector{T1}, model::choiceDDM_dx) where {T1 <: Real}

    @unpack dx, data, cross, θprior = model
    θ = Flatten.reconstruct(θchoice(), x)
    model = choiceDDM_dx(θ, data, dx, cross, θprior)
    loglikelihood(model)

end


"""
    loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
function loglikelihood(model::choiceDDM_dx)
    
    @unpack θ, data, dx, cross = model
    @unpack θz = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1].click_data

    P,M,xc,n = initialize_latent_model(σ2_i, B, λ, σ2_a, dx, dt)
    sum(pmap(data -> loglikelihood!(θ, P, M, dx, xc, data, n, cross), data))

end