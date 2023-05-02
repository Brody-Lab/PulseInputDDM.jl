"""
"""
@with_kw struct null_options
    ncells::Vector{Int}
    nparams::Int = 4
    f::String = "Sigmoid"
    fit::Vector{Bool} = repeat([true,false,false,false], sum(ncells))
    lb::Vector{Float64} = repeat([-100.,0.,-10.,-10.], sum(ncells))
    ub::Vector{Float64} = repeat([100.,100.,10.,10.], sum(ncells))
    x0::Vector{Float64} = repeat([1e-1,eps(),0.,0.], sum(ncells))
end


"""
"""
function θneural_null(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[1:nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
        
    θneural_null(θy, ncells, nparams, f)

end


"""
"""
@with_kw struct θneural_null{T1} <: DDMθ
    θy::T1
    ncells::Vector{Int}
    nparams::Int
    f::String
end


function optimize(data, options::null_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=false,
        outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural_null(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, θ)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_null(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end




"""
    flatten(θ)

Extract parameters related to the model from an defined abstract type and returns an ordered vector.
```
"""
function flatten(θ::θneural_null)

    @unpack θy = θ
    vcat(collect.(Flatten.flatten.(vcat(θy...)))...)

end


"""
    loglikelihood(x, data, θ)

A wrapper function that accepts a vector of mixed parameters and packs the parameters into an appropirate model structure
Used in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data::Union{Vector{Vector{T2}}, Vector{Any}}, 
        θ::θneural_null) where {T1 <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_null(x, ncells, nparams, f)
    loglikelihood(θ, data)

end


"""
"""
function loglikelihood(θ::θneural_null, 
        data::Union{Vector{Vector{T1}}, Vector{Any}}) where T1 <: neuraldata

    @unpack θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(θy::Vector{T1}, data::neuraldata) where T1 <: DDMf

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ = loglikelihood(θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function loglikelihood(θy::Vector{T1}, input_data::neuralinputs) where T1 <: DDMf

    @unpack λ0, dt = input_data

    λ = map((θy,λ0)-> θy(zeros(length(λ0)), λ0), θy, λ0)

end