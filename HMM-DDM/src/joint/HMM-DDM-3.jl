"""
"""
@with_kw struct HMMDDM_joint_options_3
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function HMMDDM_joint_options_3(θ::θHMMDDM_joint_3)
    
    @unpack K, f = θ
    
    nparams, ncells = nθparams(f)        
    
    fit = vcat(repeat(trues(K), K), vcat(trues(dimz), trues(2), trues.(nparams)...), 
        repeat(vcat(trues(dimz), trues(2)), K-1))
    
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
    
    lb = vcat(repeat(zeros(K), K), vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.], vcat(lb...)...), 
        repeat(vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.]), K-1))
    ub = vcat(repeat(ones(K), K), vcat([100., 40., 5., 400., 10., 1.2,  1.], [10, 1.], vcat(ub...)...),
        repeat(vcat([100., 40., 5., 400., 10., 1.2,  1.], [10, 1.]), K-1))

    HMMDDM_joint_options_3(fit=fit, ub=ub, lb=lb)
    
end
   

"""
"""
function θHMMDDM_joint_3(x::Vector{T}, θ::θHMMDDM_joint_3) where {T <: Real}
    
    @unpack K, θ, f = θ
    
    m = collect(partition(x[1:K*K], K))
    m = collect(hcat(m...))
    m = mapslices(x-> x/sum(x), m, dims=2)
    
    nparams, ncells = nθparams(f)        
        
    xyidxs = K*K+dimz+2+1: K*K+dimz+2+sum(nparams)
    xy = x[xyidxs]
    idxs = setdiff(K*K+1:length(x), xyidxs)
    xz = collect.(partition(x[idxs], Int(length(x[idxs])/K)))
    
    θHMMDDM_joint_3(θ=map(x -> θneural_choice(vcat(x, xy), f), xz), m=m, K=K, f=f)

end


"""
    flatten(θ)

Extract parameters `HMMDDM_3` model and place in the correct order into a 1D `array`
```
"""
function flatten(θ::θHMMDDM_joint_3)

    @unpack m, θ = θ

    vcat(m..., vcat(flatten(θ[1])..., getindex.(flatten.(θ[2:end]), 1:dimz+2)...))

end


"""
    optimize(model, options)

Optimize model parameters for a `HMMDDM_joint_3`.

Arguments: 

- `model`: an instance of a `HMMDDM_joint_3`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `HMMDDM_joint_3`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function optimize(model::HMMDDM_joint_3, options::HMMDDM_joint_options_3;
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
    model = HMMDDM_joint_3(θHMMDDM_joint_3(x, θ), data, n, cross, θprior)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, model)

Maps `x` into `model`. Used in optimization, Hessian and gradient computation.

Arguments:

- `x`: a vector of mixed parameters.
- `model`: an instance of `HMMDDM_joint_3`
"""
function loglikelihood(x::Vector{T}, model::HMMDDM_joint_3; remap::Bool=false, useIDs=nothing) where {T <: Real}
    
    @unpack data,θ,n,cross,θprior = model
    
    if remap        
        θ2 = θHMMDDM_joint_3(x, θ)
        model = HMMDDM_joint_3(θHMMDDM_joint_3(θ=θexp.(θ2.θ), m=θ2.m, K=θ2.K, f=θ2.f), 
            data, n, cross, θprior)
    else
        model = HMMDDM_joint_3(θHMMDDM_joint_3(x, θ), data, n, cross, θprior)
    end
    
    loglikelihood(model; useIDs=useIDs)

end