const dimz_th = 8

"""
"""
@with_kw struct th_options
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    f::Union{Vector{String}, Vector{Vector{String}}}
    fit::Vector{Bool}
    ub::Vector{Float64}
    x0::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
@with_kw struct θneural_th{T1, T2, T3} <: DDMθ
    θz::T1
    μ0::T3
    θy::T2
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    f::Union{Vector{String}, Vector{Vector{String}}}
end



"""
"""
function θneural_th(x::Vector{T}, ncells::Vector{Int}, nparams::Vector{Vector{Int}}, 
        f::Vector{Vector{String}}) where {T <: Real}
    
    borg = vcat(dimz_th, dimz_th.+cumsum(vcat(nparams...)))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_th(θz(x[1:dimz]...), x[dimz_th], θy, ncells, nparams, f)

end



"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural_th)

    @unpack θy, θz, μ0 = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, μ0, 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


function optimize(data, options::T1; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, σ::Vector{Float64}=[0.], 
        μ::Vector{Float64}=[0.], do_prior::Bool=false, cross::Bool=false,
        sig_σ::Float64=1.) where T1 <: th_options

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural_th(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, θ; n=n, cross=cross) + sigmoid_prior(stack(x,c,fit), data, θ; sig_σ=sig_σ))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_th(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


function sigmoid_prior(x::Vector{T1}, data::Union{Vector{Vector{T2}}, Vector{Any}}, 
        θ::θneural_th; sig_σ::Float64=1.) where {T1 <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_th(x, ncells, nparams, f)
    
    if typeof(f) == String
        if f == "Sigmoid"
            sum(map(x-> sum(logpdf.(Normal(0., sig_σ), map(x-> x.c, x))), θ.θy))
        else
            0.
        end
    else    
        sum(map(x-> sum(logpdf.(Normal(0., sig_σ), x.c)), vcat(θ.θy...)[vcat(f...) .== "Sigmoid"]))
    end
    
end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, data::Vector{Vector{T2}}, 
        θ::θneural_th; n::Int=53, 
        cross::Bool=false) where {T <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_th(x, ncells, nparams, f)
    loglikelihood(θ, data; n=n, cross=cross)

end


"""
"""
function initialize_latent_model(σ2_i::TT, B::TT, λ::TT, σ2_a::TT,
    μ0::VV, n::Int, dt::Float64; lapse::UU=0.) where {TT,UU,VV <: Any}

    xc,dx = bins(B,n)
    P = P0(σ2_i,μ0,n,dx,xc,dt; lapse=lapse)
    M = transition_M(σ2_a*dt,λ,zero(TT),dx,xc,n,dt)

    return P, M, xc, dx

end


"""
    P0(σ2_i, n dx, xc, dt; lapse=0.)

"""
function P0(σ2_i::TT, μ0::WW, n::Int, dx::VV, xc::Vector{TT}, dt::Float64;
    lapse::UU=0.) where {TT,UU,VV,WW <: Any}

    P = zeros(TT,n)
    P[ceil(Int,n/2)] = one(TT) - lapse
    P[1], P[n] = lapse/2., lapse/2.
    M = transition_M(σ2_i,zero(TT),μ0,dx,xc,n,dt)
    P = M * P

end


"""
    LL_all_trials(pz, py, data; n=53)

Computes the log likelihood for a set of trials consistent with the observed neural activity on each trial.
"""
function loglikelihood(θ::θneural_th, 
        data::Vector{Vector{T1}}; n::Int=53, cross::Bool=false) where {T1 <: neuraldata}

    @unpack θz, θy, μ0 = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data
   
    choice = map(x-> map(x-> x.choice, x), data)

    P0,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, zero(μ0), n, dt)
    P0_L, = initialize_latent_model(σ2_i, B, λ, σ2_a, -μ0, n, dt)
    P0_R, = initialize_latent_model(σ2_i, B, λ, σ2_a, μ0, n, dt)

    sum(map((data, θy, choice) -> sum(pmap(data -> 
                    loglikelihood(θz,θy,data, P0_R, M, xc, dx; n=n, cross=cross), 
                    data[2:end][choice[1:end-1]])), data, θy, choice)) + 
    sum(map((data, θy, choice) -> sum(pmap(data -> 
                    loglikelihood(θz,θy,data, P0_L, M, xc, dx; n=n, cross=cross), 
                    data[2:end][.!choice[1:end-1]])), data, θy, choice)) + 
    sum(map((data, θy) -> sum(pmap(data -> 
                    loglikelihood(θz,θy,data, P0, M, xc, dx; n=n, cross=cross), [data[1]])), data, θy))

end