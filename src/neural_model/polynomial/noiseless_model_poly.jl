"""
"""
function optimize(data, options::neural_poly_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.)

    @unpack fit, lb, ub, x0, ncells, f, nparams, npolys = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    #ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, ncells, nparams, f, npolys)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, ncells, nparams, f, npolys) -
        α1 * (x[2] - lb[2]).^2)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_poly(x, ncells, nparams, f, npolys)
    model = neural_poly_DDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data::Vector{Vector{T2}}, ncells::Vector{Int}, 
        nparams::Int, f::String, npolys::Int) where {T1 <: Real, T2 <: neuraldata}

    θ = θneural_poly(x, ncells, nparams, f, npolys)
    loglikelihood(θ, data)

end


"""
    gradient(model)
"""
function gradient(model::neural_poly_DDM)

    @unpack θ, data = model
    @unpack ncells, nparams, f, npolys = θ
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, npolys)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
"""
function loglikelihood(θ::θneural_poly, data::Vector{Vector{T1}}) where T1 <: neuraldata

    @unpack θz, θμ, θy = θ

    sum(map((θy, θμ, data) -> sum(pmap(data-> loglikelihood(θz, θμ, θy, data), data,
        batch_size=length(data))), θy, θμ, data))

end



"""
"""
function loglikelihood(θz::θz, θμ::Vector{Poly{T2}}, θy::Vector{T1}, 
        data::neuraldata) where {T1 <: DDMf, T2 <: Real}

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = loglikelihood(θz,θμ,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function loglikelihood(θz::θz, θμ::Vector{Poly{T2}}, θy::Vector{T1}, 
        input_data::neuralinputs) where {T1 <: DDMf, T2 <: Real}

    @unpack binned_clicks, dt = input_data
    @unpack nT = binned_clicks

    a = rand(θz,input_data)
    λ = map((θy,θμ)-> θy(a, θμ(1:nT)), θy, θμ)

    return λ, a

end