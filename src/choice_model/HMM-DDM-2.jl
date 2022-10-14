"""
    save_model(file, model, options)

Given a `file`, `model` and `options` produced by `optimize`, save everything to a `.MAT` file in such a way that `reload_neural_data` can bring these things back into a Julia workspace, or they can be loaded in MATLAB.

See also: [`reload_neural_model`](@ref)

"""
function save_model(file, model::Union{HMMDDM_choice_2}, options)

    @unpack lb, ub, fit = options
    @unpack θ, data, n, cross = model
    @unpack K = θ
        
    dict = Dict("ML_params"=> collect(pulse_input_DDM.flatten(θ)),
        "lb"=> lb, "ub"=> ub, "fit"=> fit, "n"=> n, "cross"=> cross,
        "K" => K)

    matwrite(file, dict)

end


"""
"""
@with_kw struct HMMDDM_choice_options_2
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function HMMDDM_choice_options_2(θ::θHMMDDM_choice_2)
    
    @unpack K = θ
        
    fit = vcat(repeat(trues(K), K), repeat(vcat(trues(dimz), trues(2)), K))
    lb = vcat(repeat(zeros(K), K), repeat(vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-5, 0.]), K))
    ub = vcat(repeat(ones(K), K), repeat(vcat([100., 40., 5., 400., 10., 1.2,  1.], [5, 1.]), K))

    HMMDDM_choice_options_2(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θHMMDDM_choice_2(x::Vector{T}, θ::θHMMDDM_choice_2) where {T <: Real}
    
    @unpack K, θ = θ
    
    m = collect(partition(x[1:K*K], K))
    m = collect(hcat(m...))
    m = mapslices(x-> x/sum(x), m, dims=2)
        
    xz = collect.(partition(x[K*K+1:end], Int(length(x[K*K+1:end])/K)))
    
    θHMMDDM_choice_2(θ=map(x -> Flatten.reconstruct(θchoice(), x), xz), m=m, K=K)

end


"""
    flatten(θ)

Extract parameters `HMMDDM_2` model and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θHMMDDM_choice_2)

    @unpack m, θ = θ

    vcat(m..., vcat(collect.(Flatten.flatten.(θ))...))

end


"""
    optimize(model, options)

Optimize model parameters for a `HMMDDM_choice_2`.

Arguments: 

- `model`: an instance of a `HMMDDM_choice_2`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `HMMDDM_choice_2`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::HMMDDM_choice_2, options::HMMDDM_choice_options_2;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1), 
        scaled::Bool=false, extended_trace::Bool=false, useIDs=nothing)
    
    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross, θprior = model
    
    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0, c = unstack(x0, fit)
    
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model; useIDs=useIDs)
    
    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = HMMDDM_choice_2(θHMMDDM_choice_2(x, θ), data, n, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `HMMDDM_choice_2`
"""
function loglikelihood(x::Vector{T}, model::HMMDDM_choice_2; remap::Bool=false, useIDs=nothing) where {T <: Real}
    
    @unpack data,θ,n,cross,θprior = model
    
    if remap        
        θ2 = HMMDDM_choice_2(x, θ)
        model = HMMDDM_choice_2(θHMMDDM_choice_2(θ=θexp.(θ2.θ), m=θ2.m, K=θ2.K), 
            data, n, cross, θprior)
    else
        model = HMMDDM_choice_2(θHMMDDM_choice_2(x, θ), data, n, cross, θprior)
    end
    
    loglikelihood(model; useIDs=useIDs)

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `HMMDDM_choice_2`.

Arguments:

- `model`: instance of `HMMDDM_choice_2`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
"""
function Hessian(model::HMMDDM_choice_2; chunk_size::Int=4, remap=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `HMMDDM_choice_2`.
"""
function gradient(model::HMMDDM_choice_2)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
    loglikelihood(model)

Arguments: `HMMDDM_choice_2` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood(model::HMMDDM_choice_2; useIDs=nothing)
    
    @unpack data,θ,n,cross = model
    
    ntrials = length.(data)
    if isnothing(useIDs)
        useIDs = map(ntrials -> 1:ntrials, ntrials)
    end
        
    use_data = map((data, useIDs)-> data[useIDs], data, useIDs)
    
    LL = map(θ-> map(use_data-> log.(likelihood(choiceDDM(θ=θ, data=use_data, n=n, cross=cross))), use_data), θ.θ)     
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    
    LL2 = zeros.(typeof(py[1][1]), ntrials, θ.K)
    map((LL2, useIDs, py)-> LL2[useIDs, :] = py, LL2, useIDs, py)
    sum(pmap(LL2-> loglikelihood(LL2, θ)[1], LL2))

end


"""
    posterior(model)

Arguments: `HMMDDM_choice_2` instance

Returns: posterior of the data given the parameters.
"""
function posterior(model::HMMDDM_choice_2; useIDs=nothing)
    
    @unpack data,θ,n,cross = model
    
    ntrials = length.(data)
    if isnothing(useIDs)
        useIDs = map(ntrials -> 1:ntrials, ntrials)
    end
        
    use_data = map((data, useIDs)-> data[useIDs], data, useIDs)
    
    LL = map(θ-> map(use_data-> log.(likelihood(choiceDDM(θ=θ, data=use_data, n=n, cross=cross))), use_data), θ.θ) 
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    
    LL2 = zeros.(typeof(py[1][1]), ntrials, θ.K)
    map((LL2, useIDs, py)-> LL2[useIDs, :] = py, LL2, useIDs, py)
    
    #sum(pmap(LL2-> loglikelihood(LL2, θ)[1], LL2))
    #pmap(py-> posterior(py, θ), py)
    pmap(LL2-> posterior(LL2, θ), LL2)

end