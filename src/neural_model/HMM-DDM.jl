"""
"""
@with_kw struct θHMMDDM{T1,T2,T3} <: DDMθ
    θz::Vector{T1}
    θy::T2
    f::Vector{Vector{String}}
    m::Array{T3,2}=[0.2 0.8; 0.1 0.9]
    K::Int=2
end


"""
    HMMDDM

Fields:
- θ
- data
- n
- cross
- θprior

"""
@with_kw struct HMMDDM{U,V} <: DDM
    θ::θHMMDDM
    data::U
    n::Int=53
    cross::Bool=false
    θprior::V = θprior()
end


"""
"""
@with_kw struct HMMDDM_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
    save_model(file, model, options)

Given a `file`, `model` and `options` produced by `optimize`, save everything to a `.MAT` file in such a way that `reload_neural_data` can bring these things back into a Julia workspace, or they can be loaded in MATLAB.

See also: [`reload_neural_model`](@ref)

"""
function save_model(file, model::HMMDDM, options)

    @unpack lb, ub, fit = options
    @unpack θ, data, n, cross = model
    @unpack f, K = θ
    @unpack dt, delay, pad = data[1][1].input_data
    
    nparams, ncells = nθparams(f)
    
    dict = Dict("ML_params"=> collect(pulse_input_DDM.flatten(θ)),
        "lb"=> lb, "ub"=> ub, "fit"=> fit, "n"=> n, "cross"=> cross,
        "dt"=> dt, "delay"=> delay, "pad"=> pad, "f"=> vcat(vcat(f...)...),
        "nparams" => nparams, "ncells" => ncells, "K" => K)

    matwrite(file, dict)

end


"""
"""
function HMMDDM_options(θ::θHMMDDM)
    
    @unpack K, f = θ
    
    nparams, ncells = nθparams(f)        
    
    fit = vcat(repeat(vcat(trues(K-1), falses(1)), K), trues(K*dimz), trues.(nparams)...)
    
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
    
    lb = vcat(repeat(zeros(K), K), repeat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], K), vcat(lb...))
    ub = vcat(repeat(ones(K), K),  repeat([100., 100., 5., 400., 10., 1.2,  1.], K), vcat(ub...));

    HMMDDM_options(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θHMMDDM(x::Vector{T}, θ::θHMMDDM) where {T <: Real}
    
    @unpack K, f = θ
    
    m = reshape(x[1:K*(K-1)], K, K-1)
    summ = sum(m, dims=2)
    m = hcat(m, 1 .- summ)
    
    xz = collect(partition(x[K*K+1:K*K+K*dimz], dimz))  
    
    idx = K*K+K*dimz     
    nparams, ncells = nθparams(f)   
    borg = vcat(idx,idx.+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]   
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)  
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θHMMDDM(θz=map(x -> θz(x...), xz), θy=θy, f=f, m=m, K=K)

end


"""
    flatten(θ)

Extract parameters `HMMDDM` model and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θHMMDDM)

    @unpack m, θz, θy = θ

    vcat(m..., vcat(collect.(Flatten.flatten.(θz))...), 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    optimize(model, options)

Optimize model parameters for a `HMMDDM`.

Arguments: 

- `model`: an instance of a `HMMDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `HMMDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::HMMDDM, options::HMMDDM_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross, θprior = model
    
    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0, c = unstack(x0, fit)
    
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model)[1]
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = HMMDDM(θHMMDDM(x, θ), data, n, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `HMMDDM`
"""
function loglikelihood(x::Vector{T}, model::HMMDDM) where {T <: Real}
    
    @unpack data,θ,n,cross,θprior = model
    model = HMMDDM(θHMMDDM(x, θ), data, n, cross, θprior)

    loglikelihood(model)

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `HMMDDM`.

Arguments:

- `model`: instance of `HMMDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
"""
function Hessian(model::HMMDDM; chunk_size::Int=4)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    loglikelihood(model)

Arguments: `HMMDDM` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood(model::HMMDDM)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy, f = θ
    @unpack m, K = θ
    
    LL = map(θz-> exp.(vcat(loglikelihood_pertrial(neuralDDM(θ=θneural(θz=θz, θy=θy, f=f), data=data, n=n, cross=cross))...)), θz)
    
    c = Vector(undef, length(LL[1]))
    ps = Vector(undef, length(LL[1]))
    p = 1/K * ones(K)
    
    @inbounds for t = 1:length(c)

        p = m * p    
        p = p .* vec(getindex.(LL, t))     
        c[t] = sum(p)
        p /= c[t]
        ps[t] = p

    end

    return sum(log.(c)), ps

end

#=
"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `neuralDDM` or a `noiseless_neuralDDM`.
"""
function gradient(model::Union{neuralDDM, noiseless_neuralDDM})

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
"""
function train_and_test(data; 
        n::Int=53, cross::Bool=false,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false,
        x0_z::Vector{Float64}=[0.1, 15., -0.1, 20., 0.8, 0.01, 0.008],
        seed::Int=1, σ_B::Float64=1e6, sig_σ::Float64=1.)
    
    ncells = getfield.(first.(data), :ncells)
    f = repeat(["Softplus"], sum(ncells))
    borg = vcat(0,cumsum(ncells))
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]];
        
    ntrials = length.(data)
    train = map(ntrials -> sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false), ntrials)
    test = map((ntrials, train)-> setdiff(1:ntrials, train), ntrials, train)
    
    model, options = optimize(map((data, train)-> data[train], data, train), f; 
        n=n, cross=cross,
        x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, 
        iterations=iterations, show_trace=show_trace, 
        outer_iterations=outer_iterations, extended_trace=extended_trace, 
        scaled=scaled, sig_σ=sig_σ, x0_z=x0_z, 
        θprior=θprior(μ_B=40., σ_B=σ_B))
        
    testLL = loglikelihood(neuralDDM(model.θ, map((data, test)-> data[test], data, test), n, cross, θprior(μ_B=40., σ_B=σ_B)))
    LL = loglikelihood(neuralDDM(model.θ, data, n, cross, θprior(μ_B=40., σ_B=σ_B)))

    return σ_B, model, testLL, LL, options
    
end

=#