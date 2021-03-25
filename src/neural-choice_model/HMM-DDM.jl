"""
"""
@with_kw struct HMMDDM_joint_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function HMMDDM_joint_options(θ::θHMMDDM_joint)
    
    @unpack K, f = θ
    
    nparams, ncells = nθparams(f)        
    
    fit = vcat(repeat(trues(K), K), trues(K*dimz), trues(2), trues.(nparams)...)
    
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
    
    lb = vcat(repeat(zeros(K), K), repeat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], K), [-10, 0.], vcat(lb...))
    ub = vcat(repeat(ones(K), K),  repeat([100., 40., 5., 400., 10., 1.2,  1.], K), [10, 1.], vcat(ub...));

    HMMDDM_joint_options(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θHMMDDM_joint(x::Vector{T}, θ::θHMMDDM_joint) where {T <: Real}
    
    @unpack K, f = θ
    
    m = collect(partition(x[1:K*K], K))
    m = map(m-> m/sum(m), m)
    m = collect(hcat(m...)')
    
    xz = collect(partition(x[K*K+1:K*K+K*dimz], dimz))  
    idx = K*K+K*dimz
    
    bias,lapse = x[idx+1], x[idx+2]
    
    idx = K*K+K*dimz+2     
    nparams, ncells = nθparams(f)   
    borg = vcat(idx,idx.+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]   
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)  
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θHMMDDM_joint(θz=map(x -> θz(x...), xz), bias=bias, lapse=lapse, θy=θy, f=f, m=m, K=K)

end


"""
    flatten(θ)

Extract parameters `HMMDDM` model and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θHMMDDM_joint)

    @unpack m, θz, θy, bias, lapse = θ

    vcat(m..., vcat(collect.(Flatten.flatten.(θz))...), 
        bias, lapse, vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    optimize(model, options)

Optimize model parameters for a `HMMDDM_joint`.

Arguments: 

- `model`: an instance of a `HMMDDM_joint`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `HMMDDM_joint`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::HMMDDM_joint, options::HMMDDM_joint_options;
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
    model = HMMDDM_joint(θHMMDDM_joint(x, θ), data, n, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `HMMDDM_joint`
"""
function loglikelihood(x::Vector{T}, model::HMMDDM_joint) where {T <: Real}
    
    @unpack data,θ,n,cross,θprior = model
    model = HMMDDM_joint(θHMMDDM_joint(x, θ), data, n, cross, θprior)

    loglikelihood(model)

end


"""
    Hessian(model; chunck_size)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `HMMDDM_joint`.

Arguments:

- `model`: instance of `HMMDDM_joint`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
"""
function Hessian(model::HMMDDM_joint; chunk_size::Int=4)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end


"""
    gradient(model)

Compute the gradient of the negative log-likelihood at the current value of the parameters of a `HMMDDM_joint`.
"""
function gradient(model::HMMDDM_joint)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)::Vector{Float64}

end


"""
"""
function logsumexp(x)
    m = maximum(x)
    m + log(sum(exp.(x .- m)))
end


"""
    loglikelihood(model)

Arguments: `HMMDDM_joint` instance

Returns: loglikehood of the data given the parameters.
"""
function loglikelihood(model::HMMDDM_joint)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy, bias, lapse, f = θ
        
    LL = map(θz-> joint_loglikelihood_per_trial(neural_choiceDDM(
        θ=θneural_choice(θz=θz, bias=bias, lapse=lapse, θy=θy, f=f), data=data, n=n, cross=cross)), θz)  
    #py = map(i-> hcat(map(k-> max.(1e-150, exp.(LL[k][i])), 1:length(LL))...), 1:length(LL[1]))  
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    sum(pmap(py-> loglikelihood(py, θ)[1], py))

end


"""
    posterior(model)

Arguments: `HMMDDM_joint` instance

Returns: posterior of the data given the parameters.
"""
function posterior(model::HMMDDM_joint)
    
    @unpack data,θ,n,cross = model
    @unpack θz, θy, f, bias, lapse = θ
    
    LL = map(θz-> joint_loglikelihood_per_trial(neural_choiceDDM(
        θ=θneural_choice(θz=θz, bias=bias, lapse=lapse, θy=θy, f=f), data=data, n=n, cross=cross)), θz)  
    #py = map(i-> hcat(map(k-> max.(1e-150, exp.(LL[k][i])), 1:length(LL))...), 1:length(LL[1]))  
    py = map(i-> hcat(map(k-> LL[k][i], 1:length(LL))...), 1:length(LL[1]))  
    pmap(py-> posterior(py, θ), py)

end


#=
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