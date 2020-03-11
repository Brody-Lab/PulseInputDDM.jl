abstract type neural_options end


"""
"""
function train_and_test(data, options::neural_options, n; seed::Int=1, α1s = 10. .^(-6:7))
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    model = map(α1-> optimize([data[train]], options, n; α1=α1, show_trace=false)[1], α1s)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]], n), model)

    return α1s, model, testLL
    
end


"""
"""
@with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    ncells::Vector{Int}
    nparams::Int
    f::String
end


"""
"""
@with_kw struct Sigmoid{T1} <: DDMf
    a::T1=10.
    b::T1=10.
    c::T1=1.
    d::T1=0.
end


"""
"""
(θ::Sigmoid)(x::Vector{U}, λ0::Vector{T}) where {U,T <: Real} =
    (θ::Sigmoid).(x, λ0)


"""
"""
function (θ::Sigmoid)(x::U, λ0::T) where {U,T <: Real}

    @unpack a,b,c,d = θ

    y = c * x + d
    y = a + b * logistic!(y)
    y = softplus(y + λ0)
    #y = max(eps(), y + λ0)

end


"""
"""
@with_kw struct Softplus{T1} <: DDMf
    a::T1 = 10.
    c::T1 = 5.0*rand([-1,1])
    d::T1 = 0
end


"""
"""
function (θ::Softplus)(x::Union{U,Vector{U}}, λ0::Union{T,Vector{T}}) where {U,T <: Real}

    @unpack a,c,d = θ

    y = a .+ softplus.(c*x .+ d .+ λ0)
    #y = max.(eps(), y .+ λ0)
    #y = softplus.(y .+ λ0)

end


"""
"""
@with_kw struct θy{T1}
    θ::T1
end


"""
"""
@with_kw struct neuraldata <: DDMdata
    input_data::neuralinputs
    spikes::Vector{Vector{Int}}
    ncells::Int
end


"""
"""
@with_kw struct neuralDDM{T,U} <: DDM
    θ::T
    data::U
end


"""
"""
@with_kw struct Sigmoid_options <: neural_options
    ncells::Vector{Int}
    nparams::Int = 4
    f::String = "Sigmoid"
    fit::Vector{Bool} = vcat(trues(dimz + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005],
        repeat([-100.,0.,-10.,-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, Inf, 10., Inf, Inf, 1.2,  1.],
        repeat([100.,100.,10.,10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008],
        repeat([10.,10.,1.,0.], sum(ncells)))
end


"""
"""
@with_kw struct Softplus_options <: neural_options
    ncells::Vector{Int}
    nparams::Int = 3
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(trues(dimz + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -10., 0.,   0.,  0., 0.005],
        repeat([1e-12, -10., -10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, 200., 10., Inf, Inf, 1.2,  1.],
        repeat([100., 10., 10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008],
        repeat([10.,1.,0.], sum(ncells)))
end


"""
"""
neuraldata(input_data, spikes::Vector{Vector{Vector{Int}}}, ncells::Int) =  neuraldata.(input_data,spikes,ncells)


"""
"""
function θneural(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[dimz + 1:dimz + nparams*sum(ncells)], nparams)))
    
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
function loglikelihood(x::Vector{T}, data::Vector{Vector{T2}}, θ::θneural; n::Int=53) where {T <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural(x, ncells, nparams, f)
    loglikelihood(θ, data; n=n)

end


"""
    gradient(model; n=53)
"""
function gradient(model::neuralDDM, n::Int)

    @unpack θ, data = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, data, θ; n=n)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    Hessian(model; n=53)
"""
function Hessian(model::neuralDDM, n::Int; chuck_size::Int=4)

    @unpack θ, data = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, data, θ; n=n)

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
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    optimize_model(pz, py, data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

BACK IN THE DAY, TOLS USED TO BE x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize(data, options::T1; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, α1::Float64=0.) where T1 <: neural_options

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    #ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, ncells, nparams, f, npolys, n)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, θ; n=n) -
        α1 * (x[2] - lb[2]).^2)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    LL_all_trials(pz, py, data; n=53)

Computes the log likelihood for a set of trials consistent with the observed neural activity on each trial.
"""
function loglikelihood(θ::θneural, data::Vector{Vector{T1}}; n::Int=53) where {T1 <: neuraldata}

    @unpack θz, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    sum(map((data, θy) -> sum(pmap(data -> 
                    loglikelihood(θz,θy,data,P, M, xc, dx; n=n), data)), data, θy))

end


"""
"""
function loglikelihood(θz,θy,data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3; n::Int=53) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R)

    c = Vector{T1}(undef,nT)
    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:nT

        if centered && t == 1
            P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt/2)
        else
            P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        end

        P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                                k[t]), spikes, θy, λ0))), xc)...)
        c[t] = sum(P)
        P /= c[t]

    end

    return sum(log.(c))

end
