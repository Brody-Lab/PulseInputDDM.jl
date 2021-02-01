"""
"""
@with_kw struct HMMDDM_joint_options_2
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function HMMDDM_joint_options_2(θ::θHMMDDM_joint_2)
    
    @unpack K, f = θ
    
    nparams, ncells = nθparams(f)        
    
    fit = vcat(repeat(trues(K), K), repeat(vcat(trues(dimz), trues(2), trues.(nparams)...), K))
    
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
    
    lb = vcat(repeat(zeros(K), K), repeat(vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.], vcat(lb...)...), K))
    ub = vcat(repeat(ones(K), K), repeat(vcat([100., 40., 5., 400., 10., 1.2,  1.], [10, 1.], vcat(ub...)...), K))

    HMMDDM_joint_options_2(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θHMMDDM_joint_2(x::Vector{T}, θ::θHMMDDM_joint_2) where {T <: Real}
    
    @unpack K, θ, f = θ
    
    m = collect(partition(x[1:K*K], K))
    #m = map(m-> m/sum(m), m)
    #m = collect(hcat(m...)')
    m = collect(hcat(m...))
    m = mapslices(x-> x/sum(x), m, dims=2)
        
    xz = collect.(partition(x[K*K+1:end], Int(length(x[K*K+1:end])/K)))
    
    θHMMDDM_joint_2(θ=map(x -> θneural_choice(x, f), xz), m=m, K=K, f=f)

end


"""
    flatten(θ)

Extract parameters `HMMDDM_2` model and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θHMMDDM_joint_2)

    @unpack m, θ = θ

    vcat(m..., vcat(flatten.(θ)...))

end


"""
    optimize(model, options)

Optimize model parameters for a `HMMDDM_joint_2`.

Arguments: 

- `model`: an instance of a `HMMDDM_joint_2`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `HMMDDM_joint_2`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::HMMDDM_joint_2, options::HMMDDM_joint_options_2;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross, θprior = model
    
    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0, c = unstack(x0, fit)
    
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = HMMDDM_joint_2(θHMMDDM_joint_2(x, θ), data, n, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `HMMDDM_joint_2`
"""
function loglikelihood(x::Vector{T}, model::HMMDDM_joint_2; remap::Bool=false) where {T <: Real}
    
    @unpack data,θ,n,cross,θprior = model
    
    if remap        
        θ2 = θHMMDDM_joint_2(x, θ)
        model = HMMDDM_joint_2(θHMMDDM_joint_2(θ=θexp.(θ2.θ), m=θ2.m, K=θ2.K, f=θ2.f), 
            data, n, cross, θprior)
    else
        model = HMMDDM_joint_2(θHMMDDM_joint_2(x, θ), data, n, cross, θprior)
    end
    
    loglikelihood(model)

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `HMMDDM_joint_2`.

Arguments:

- `model`: instance of `HMMDDM_joint_2`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
"""
function Hessian(model::HMMDDM_joint_2; chunk_size::Int=4, remap=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `HMMDDM_joint_2`.
"""
function gradient(model::HMMDDM_joint_2)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    loglikelihood(model)

Arguments: `HMMDDM_joint_2` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood(model::HMMDDM_joint_2)
    
    @unpack data,θ,n,cross = model
        
    LL = map(θ-> joint_loglikelihood_per_trial(neural_choiceDDM(θ=θ, data=data, n=n, cross=cross)), θ.θ)  
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    sum(pmap(py-> loglikelihood(py, θ)[1], py))

end


"""
    posterior(model)

Arguments: `HMMDDM_joint_2` instance

Returns: posterior of the data given the parameters.
"""
function posterior(model::HMMDDM_joint_2)
    
    @unpack data,θ,n,cross = model
    
    LL = map(θ-> joint_loglikelihood_per_trial(neural_choiceDDM(θ=θ, data=data, n=n, cross=cross)), θ.θ) 
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    pmap(py-> posterior(py, θ), py)

end