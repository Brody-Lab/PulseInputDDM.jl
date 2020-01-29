"""
"""
@flattenable @with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1 = θz() | true
    θy::T2 | true
    ncells::Vector{Int} | false
    nparams::Int
    f::String
end


"""
"""
@with_kw struct Sigmoid{T1}
    a::T1=10.
    b::T1=10.
    c::T1=1.
    d::T1=0.
end


"""
"""
(θ::Sigmoid)(x::Vector{U}, λ0::Vector{Float64}) where U <: Real =
    (θ::Sigmoid).(x, λ0)


"""
"""
function (θ::Sigmoid)(x::U, λ0::Float64) where U <: Real

    @unpack a,b,c,d = θ

    y = c * x + d
    y = a + b * logistic!(y)
    y = softplus(y + λ0)

end


"""
"""
@with_kw struct Softplus{T1}
    a::T1 = 10.
    c::T1 = 5.0*rand([-1,1])
    d::T1 = 0
end


"""
"""
function (θ::Softplus)(x::Union{U,Vector{U}}, λ0::Union{Float64,Vector{Float64}}) where U <: Real

    @unpack a,c,d = θ

    y = a .+ softplus.(c*x .+ d)
    #y = max.(eps(), y .+ λ0)
    y = softplus.(y .+ λ0)

end


"""
"""
@with_kw struct θy{T1}
    θ::T1
end


@with_kw struct neuraldata <: DDMdata
    input_data::neuralinputs
    spikes::Vector{Vector{Int}}
    ncells::Int
end


"""
"""
@with_kw struct neuralDDM{T,U} <: DDM
    θ::T = θneural()
    data::U
end


"""
"""
neuraldata(input_data, spikes::Vector{Vector{Vector{Int}}}, ncells::Int) =  neuraldata.(input_data,spikes,ncells)


"""
"""
function unflatten(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}

    #this is hardcoded for sig
    #not very good, has to be a better way to split into 2 and 3
    dims2 = vcat(0,cumsum(ncells))
    blah = Tuple.(collect(partition(x[dimz+1:end], nparams)))
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    θneural(θz(Tuple(x[1:dimz])...), θy, ncells, nparams, f)

end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, data, ncells::Vector{Int}, nparams, f, n::Int) where {T <: Real}

    θ = unflatten(x, ncells, nparams, f)
    loglikelihood(θ, data, n)

end


"""
    gradient(model; n=53)
"""
function gradient(model::neuralDDM, n::Int)

    @unpack θ, data = model
    @unpack ncells, nparams, f = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, n)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    Hessian(model; n=53)
"""
function Hessian(model::neuralDDM, n::Int; chuck_size::Int=4)

    @unpack θ, data = model
    @unpack ncells, nparams, f = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, n)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chuck_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural)

    @unpack θy, θz = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, vcat(collect.(Flatten.flatten.(vcat(θ.θy...)))...))

end


"""
"""
function initialize_py!(pz, py, data, f_str; show_trace::Bool=false)

    pztemp = deepcopy(pz)
    pztemp["fit"] = falses(dimz)
    pztemp["initial"][[1,4,5]] .= 2*eps()

    py["initial"] = map(data-> regress_init(data, f_str), data)
    pztemp, py = optimize_model_deterministic(pztemp, py, data, f_str, show_trace=show_trace)
    delete!(py,"final")

    return py

end


"""
    optimize_model(pz, py, data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

BACK IN THE DAY, TOLS USED TO BE x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize(data, options::neuraloptions, n::Int;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, x0, ncells, f, nparams = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, ncells, nparams, f, n)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = unflatten(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model

end
