"""
"""
@with_kw struct neural_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end



"""
"""
function neural_options(f)
    
    nparams, ncells = nθparams(f)
    fit = vcat(trues(dimz), trues.(nparams)...)
        
    lb = Vector(undef, sum(ncells))
    ub = Vector(undef, sum(ncells))
    
    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb[i] = [-10]
            ub[i] = [10]
        elseif vcat(f...)[i] == "Sigmoid"
            lb[i] = [-100.,0.,-10.,-10.]
            ub[i] = [100.,100.,10.,10.]
        end
    end
    lb = vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], vcat(lb...))
    ub = vcat([100., 100., 5., 400., 10., 1.2,  1.], vcat(ub...));

    neural_options(fit=fit, ub=ub, lb=lb)
    
end


"""
"""
@with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end
   

"""
"""
function θneural(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(dimz,dimz.+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural(θz(x[1:dimz]...), θy, f)

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
    choice::Bool
end


"""
"""
neuraldata(input_data, spikes::Vector{Vector{Vector{Int}}}, ncells::Int, choice) =  neuraldata.(input_data,spikes,ncells,choice)


"""
"""
@with_kw struct neuralDDM{T,U} <: DDM
    θ::T
    data::U
    n::Int
    cross::Bool
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
    y = a + b * logistic!(y) + λ0
    y = softplus(y)

end


"""
   Softplus(c)

``\\lambda(a) = \\ln(1 + \\exp(c * a))``
"""
@with_kw struct Softplus{T1} <: DDMf
    #a::T1 = 0
    c::T1 = 5.0*rand([-1,1])
end


"""
"""
function (θ::Softplus)(x::Union{U,Vector{U}}, λ0::Union{T,Vector{T}}) where {U,T <: Real}

    #@unpack a,c = θ
    @unpack c = θ

    #y = a .+ softplus.(c*x .+ d) .+ λ0
     #y = softplus.(c*x .+ a .+ λ0)
     y = softplus.(c*x .+ softplusinv.(λ0))
    #y = max.(eps(), y .+ λ0)
    #y = softplus.(y .+ λ0)
end

softplusinv(x) = log(expm1(x))


"""
"""
function nθparams(f)
    
    ncells = length.(f)
    nparams = Vector{Int}(undef, sum(ncells));    
    nparams[vcat(f...) .== "Softplus"] .= 1
    nparams[vcat(f...) .== "Sigmoid"] .= 4
    
    return nparams, ncells
    
end


"""
"""
function train_and_test(data, x0, options::neural_options; seed::Int=1, α1s = 10. .^(-3:1))
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    model = map(α1-> optimize([data[train]], x0, options; α1=α1, show_trace=false)[1], α1s)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]]), model)

    return α1s, model, testLL
    
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
    gradient(model)
"""
function gradient(model::neuralDDM)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    Hessian(model)
"""
function Hessian(model::neuralDDM; chuck_size::Int=4)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chuck_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
"""
function logprior(x,μ,σ) 
    
    if (x[2] >= μ[2])
        logpdf(Laplace(μ[2], σ[2]), x[2]) 
    else
        0.
    end

end


"""
"""
function optimize(data, f::Vector{Vector{String}}; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, cross::Bool=false,
        sig_σ::Float64=1., x0_z::Vector{Float64}=[0.1, 15., -0.1, 20., 0.8, 0.01, 0.008]) 
        
    θy0 = θy.(data, f)
    model0, = optimize(data, vcat(vcat(θy0...)...), f, neural_options_noiseless(f), show_trace=false)
       
    x0 = vcat(x0_z, pulse_input_DDM.flatten(model0.θ)[dimz+1:end]) 
    options = neural_options(f)  
    model, = optimize(data, x0, f, options; 
        n=n, cross=cross, show_trace=show_trace, f_tol=f_tol, 
        iterations=iterations, outer_iterations=outer_iterations)

    return model, options

end



"""
    optimize_model(pz, py, data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

BACK IN THE DAY, TOLS USED TO BE x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize(data, x0, f::Vector{Vector{String}}, options::neural_options; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false, cross::Bool=false, sig_σ::Float64=1.)
    
    @unpack fit, lb, ub = options
    
    θ = θneural(x0, f)
    model = neuralDDM(θ, data, n, cross)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), model) + sigmoid_prior(stack(x,c,fit), θ; sig_σ=sig_σ))
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = neuralDDM(θneural(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, model::neuralDDM) where {T <: Real}
    
    @unpack data,θ,n,cross = model
    @unpack f = θ 
    model = neuralDDM(θneural(x, f), data, n, cross)
    loglikelihood(model)

end


"""
    loglikelihood(model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(model::neuralDDM)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    sum(map((data, θy) -> sum(pmap(data -> 
                    loglikelihood(θz,θy,data, P, M, xc, dx, n, cross), data)), data, θy))

end


"""
"""
function loglikelihood(θz,θy,data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R;cross=cross)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    c = Vector{T1}(undef, length(time_bin))

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end

        #weird that this wasn't working....
        #P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
        #                        k[t]), spikes, θy, λ0))), xc)...)
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]

    end

    return sum(log.(c))

end