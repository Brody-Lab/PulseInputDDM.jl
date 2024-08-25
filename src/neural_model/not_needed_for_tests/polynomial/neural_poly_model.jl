abstract type neural_poly_options end


"""
"""
@with_kw struct Sigmoid_poly_options <: neural_poly_options
    ncells::Vector{Int}
    nparams::Int = 4
    npolys::Int = 4
    f::String = "Sigmoid"
    fit::Vector{Bool} = vcat(trues(dimz + sum(ncells)*npolys + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005],
        repeat(-Inf * ones(npolys), sum(ncells)),
        repeat([-100.,0.,-10.,-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, Inf, 10., Inf, Inf, 1.2,  1.],
        repeat(Inf * ones(npolys), sum(ncells)),
        repeat([100.,100.,10.,10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008],
        repeat(zeros(npolys), sum(ncells)),
        repeat([10.,10.,1.,0.], sum(ncells)))
end


"""
"""
@with_kw struct Softplus_poly_options <: neural_poly_options
    ncells::Vector{Int}
    nparams::Int = 3
    npolys::Int = 4
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(trues(dimz + sum(ncells)*npolys + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -10., 0.,   0.,  0., 0.005],
        repeat(-Inf * ones(npolys), sum(ncells)),
        repeat([1e-12, -10., -10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, 200., 10., Inf, Inf, 1.2,  1.],
        repeat(Inf * ones(npolys), sum(ncells)),
        repeat([100., 10., 10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008],
        repeat(zeros(npolys), sum(ncells)),
        repeat([10.,1.,0.], sum(ncells)))
end


"""
"""
mutable struct θneural_poly{T1, T2, T3} <: DDMθ
    θz::T1
    θμ::T2
    θy::T3
    ncells::Vector{Int}
    nparams::Int
    f::String
    npolys::Int
end


"""
"""
@with_kw struct θμ{T1}
    θ::T1
end


"""
"""
function θneural_poly(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String, npolys::Int) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[dimz + npolys*sum(ncells) + 1:dimz + npolys*sum(ncells) + nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
    
    blah = Tuple.(collect(partition(x[dimz+1:dimz+npolys*sum(ncells)], npolys)))
    
    blah2 = map(x-> Poly(collect(x)), blah)
    
    θμ = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    
    θneural_poly(θz(Tuple(x[1:dimz])...), θμ, θy, ncells, nparams, f, npolys)

end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, data, θ::θneural_poly; n::Int=53) where {T <: Real}

    @unpack ncells, nparams, f, npolys = θ
    θ = θneural_poly(x, ncells, nparams, f, npolys)
    loglikelihood(θ, data; n=n)

end


"""
    gradient(model; n=53)
"""
function gradient(model::neural_poly_DDM, n::Int)

    @unpack θ, data = model
    @unpack ncells, nparams, f, npolys = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, npolys, n)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    Hessian(model; n=53)
"""
function Hessian(model::neural_poly_DDM, n::Int; chuck_size::Int=4)

    @unpack θ, data = model
    @unpack ncells, nparams, f, npolys = θ
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, npolys, n)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chuck_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural_poly)

    @unpack θy, θz, θμ = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        vcat(coeffs.(vcat(θμ...))...),
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


function optimize(data, options::neural_poly_options; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, α1::Float64=10. .^0)

    @unpack fit, lb, ub, x0, ncells, f, nparams, npolys = options
    θ = θneural_poly(x0, ncells, nparams, f, npolys)

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
    θ = θneural_poly(x, ncells, nparams, f, npolys)
    model = neural_poly_DDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    LL_all_trials(pz, py, data; n=53)

Computes the log likelihood for a set of trials consistent with the observed neural activity on each trial.
"""
function loglikelihood(θ::θneural_poly, data; n::Int=53)

    @unpack θz, θμ, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    sum(map((data, θμ, θy) -> 
            sum(pmap(data -> loglikelihood(θz,θμ,θy,data,P, M, xc, dx, n=n), data)), data, θμ, θy))
    #sum(pmap((data, θy) -> loglikelihood(θz,θy,data,P, M, xc, dx, n),
    #    vcat(data...), vcat(map((x,y)-> repeat([x],y), θy, length.(data))...)))

end


"""
"""
function loglikelihood(θz,θμ,θy,data,
        P::Vector{TT}, M::Array{TT,2},
        xc::Vector{TT}, dx::VV; n::Int=53) where {TT,UU,VV <: Any}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, centered = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R)

    c = Vector{TT}(undef,nT)
    F = zeros(TT,n,n) #empty transition matrix for time bins with clicks

    @inbounds for t = 1:nT

        if centered && t == 1
            P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt/2)
        else
            P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        end

        P .*= vcat(map(xc-> exp(sum(map((k,θy,θμ)-> logpdf(Poisson(θy(xc,θμ(t)) * dt),
                                k[t]), spikes, θy, θμ))), xc)...)

        c[t] = sum(P)
        P /= c[t]

    end

    return sum(log.(c))

end
