#=
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


"""
"""
@with_kw struct μ_poly_options
    ncells::Vector{Int}
    npolys::Int = 4
    fit::Vector{Bool} = trues(sum(ncells)*npolys)
    lb::Vector{Float64} = repeat(-Inf * ones(npolys), sum(ncells))
    ub::Vector{Float64} = repeat(Inf * ones(npolys), sum(ncells))
    x0::Vector{Float64} = repeat([10. ^-i for i in 0:(npolys-1)], sum(ncells))
end


"""
"""
mutable struct θμ_poly{T1} <: DDMθ
    θμ::T1
    ncells::Vector{Int}
    npolys::Int
end



"""
"""
function optimize(data, options::μ_poly_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.)

    @unpack fit, lb, ub, x0, ncells, npolys = options
    
    θ = θμ_poly(x0, ncells, npolys)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, θ)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θμ_poly(x, ncells, npolys)
    model = neural_poly_DDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end




function loglikelihood(x::Vector{T1}, data::Vector{Vector{T2}}, θ::θμ_poly) where {T1 <: Real, T2 <: neuraldata}
    
    @unpack ncells, npolys = θ
    θ = θμ_poly(x, ncells, npolys)
    loglikelihood(θ, data)

end


"""
"""
function θμ_poly(x::Vector{T}, ncells::Vector{Int}, npolys::Int) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))
    
    blah = Tuple.(collect(partition(x, npolys)))
    
    blah2 = map(x-> Poly(collect(x)), blah)
    
    θμ = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    
    θμ_poly(θμ, ncells, npolys)

end


"""
"""
function loglikelihood(θ::θμ_poly, data::Vector{Vector{T1}}) where {T2 <: Real, T1 <: neuraldata}
    
    @unpack θμ = θ

    sum(map((θμ, data) -> sum(loglikelihood.(Ref(θμ), data)), θμ, data))

end



"""
"""
function loglikelihood(θμ::Vector{Poly{T2}}, 
        data::neuraldata) where {T2 <: Real}

    @unpack spikes, input_data = data
    @unpack binned_clicks, dt, pad = input_data
    @unpack nT = binned_clicks
    
    λ = map(θμ-> softplus.(θμ(1:nT+2*pad)), θμ) 

    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end
=#


"""
    RBF
"""


"""
"""
@with_kw struct neural_poly_DDM{T,U} <: DDM
    θ::T
    data::U
end


@with_kw struct μ_RBF_options
    ncells::Vector{Int}
    nRBFs::Int = 6
    fit::Vector{Bool} = trues(sum(ncells)*nRBFs)
    lb::Vector{Float64} = repeat(zeros(nRBFs), sum(ncells))
    ub::Vector{Float64} = repeat(Inf * ones(nRBFs), sum(ncells))
    x0::Vector{Float64} = repeat([1. for i in 0:(nRBFs-1)], sum(ncells))
end


"""
"""
mutable struct θμ_RBF{T1} <: DDMθ
    θμ::T1
    ncells::Vector{Int}
    nRBFs::Int
end



function train_and_test(data, options::μ_RBF_options; seed::Int=1, nRBFs = 2:10)
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    ncells = options.ncells; 
    model = pmap(nRBF-> optimize([data[train]], μ_RBF_options(ncells=ncells, nRBFs=nRBF); show_trace=false)[1], nRBFs)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]]), model)

    return nRBFs, model, testLL
    
end



"""
"""
function optimize(data, options::μ_RBF_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.)

    @unpack fit, lb, ub, x0, ncells, nRBFs = options
    
    θ = θμ_RBF(x0, ncells, nRBFs)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, θ)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θμ_RBF(x, ncells, nRBFs)
    model = neural_poly_DDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end




function loglikelihood(x::Vector{T1}, data::Vector{Vector{T2}}, θ::θμ_RBF) where {T1 <: Real, T2 <: neuraldata}
    
    @unpack ncells, nRBFs = θ
    θ = θμ_RBF(x, ncells, nRBFs)
    loglikelihood(θ, data)

end





"""
"""
function θμ_RBF(x::Vector{T}, ncells::Vector{Int}, nRBFs::Int) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))
    
    blah = Tuple.(collect(partition(x, nRBFs)))
    
    blah2 = map(x-> collect(x), blah)
    
    θμ = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    
    θμ_RBF(θμ, ncells, nRBFs)

end


"""
"""
function loglikelihood(θ::θμ_RBF, data::Vector{Vector{T1}}) where {T2 <: Real, T1 <: neuraldata}
    
    @unpack θμ, nRBFs = θ
    
    pad = data[1][1].input_data.pad   
    maxnT = maximum(vcat(map(data-> map(data-> data.input_data.binned_clicks.nT, data), data)...))   
    x = 1:maxnT+2*pad   
    rbf = UniformRBFE(x, nRBFs, normalize=true)     

    sum(map((θμ, data) -> sum(loglikelihood.(Ref(θμ), data, Ref(rbf))), θμ, data))

end



"""
"""
function loglikelihood(θμ::Vector{Vector{T2}}, 
        data::neuraldata, rbf) where {T2 <: Real}

    @unpack spikes, input_data = data
    @unpack binned_clicks, dt, pad = input_data
    @unpack nT = binned_clicks
    
    x = 1:nT+2*pad   
    #λ = map(θμ-> max.(0., rbf(x) * θμ), θμ)     
    #λ = map(θμ-> softplus.(rbf(x) * θμ), θμ)
    λ = map(θμ-> rbf(x) * θμ, θμ)     

    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end