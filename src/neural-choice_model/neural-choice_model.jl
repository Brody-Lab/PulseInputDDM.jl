"""
"""
@with_kw struct neural_choice_options
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function neural_choice_options(f; remap::Bool=false)

    nparams, ncells = nθparams(f)
    fit = vcat(trues(dimz+2), trues.(nparams)...)

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

    if remap
        lb = vcat([-10., 8.,  -5., -20.,   -3.,   1e-3, 0.005], [-10, 0.], vcat(lb...))
        ub = vcat([ 10., 100., 5.,  20.,    3.,   1.2,  1.],    [10, 1.],  vcat(ub...));
    else
        lb = vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.], vcat(lb...))
        ub = vcat([100., 100., 5., 400., 10., 1.2,  1.], [10, 1.], vcat(ub...));
    end

    neural_choice_options(fit=fit, ub=ub, lb=lb)

end


"""
"""
function θneural_choice(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}

    nparams, ncells = nθparams(f)

    borg = vcat(dimz + 2,dimz + 2 .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)

    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

    θneural_choice(θz(x[1:dimz]...), x[dimz+1], x[dimz+2], θy, f)

end


"""
    flatten(θ)

Extract parameters related to a `neural_choiceDDM` from an instance of `θneural_choice` and returns an ordered vector.
```
"""
function flatten(θ::θneural_choice)

    @unpack θy, θz, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    all_Softplus(data)

Returns: `array` of `array` of `string`, of all Softplus
"""
function all_Softplus(data)

    ncells = getfield.(first.(data), :ncells)
    f = repeat(["Softplus"], sum(ncells))
    borg = vcat(0,cumsum(ncells))
    f = [f[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]

end

"""
    Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `neural_choiceDDM`.

Arguments:

- `model`: instance of `neural_choiceDDM`

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
- `remap`: For considering parameters in variance of std space.

"""
function Hessian(model::neural_choiceDDM; chunk_size::Int=4, remap::Bool=false)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -joint_loglikelihood(x, model; remap=remap)

    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)

end

"""
    choice_neural_optimize(data)

Optimize model parameters for a `neural_choiceDDM`. Neural tuning parameters ([`θy`](@ref)) are initialized by fitting a the noiseless DDM model first ([`noiseless_neuralDDM`](@ref)).

Arguments:

- `data`: the output of [`load_neural_data`](@ref) with the format as described in its docstring.

Returns

- `model`: a module-defined type that organizes the `data` and parameters from the fit (as well as a few other things that are necessary for re-computing things the way they were computed here (e.g. `n`)
- `options`: some details related to the optimzation, such as which parameters were fit, and the upper and lower bounds of those parameters.

"""
function choice_neural_optimize(data;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        show_trace::Bool=true,  extended_trace::Bool=false, scaled::Bool=false,
        outer_iterations::Int=Int(1e1), iterations::Int=Int(2e3),
        n::Int=53, cross::Bool=false,
        x0_z::Vector{Float64}=[0.1, 15., -0.1, 20., 0.8, 0.01, 0.008],
        f::Vector{Vector{String}}=all_Softplus(data))

    x0 = vcat(x0_z, [0., 1e-1], collect.(Flatten.flatten.(vcat(θy0(data,f)...)))...)
    nparams, = nθparams(f)
    fit = vcat(trues(dimz), trues(2), trues.(nparams)...)
    options = neural_choice_options(f)
    options = neural_choice_options(fit=fit, lb=options.lb, ub=options.ub)

    model = neural_choiceDDM(θneural_choice(x0, f), data, n, cross)
    model, = choice_neural_optimize(model, options; show_trace=show_trace, f_tol=f_tol,
        iterations=iterations, outer_iterations=outer_iterations)

    return model, options

end


"""
    choice_optimize(model, options)

Optimize choice-related model parameters for a `neural_choiceDDM` using choice data.

Arguments:

- `model`: an instance of a `neural_choiceDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function choice_optimize(model::neural_choiceDDM, options::neural_choice_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false, remap::Bool=false)

    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
    @unpack f = θ

    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -choice_loglikelihood(stack(x,c,fit), model; remap=remap)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)

    model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
    choice_neural_optimize(model, options)

Optimize (potentially all) model parameters for a `neural_choiceDDM` using choice and neural data.

Arguments:

- `model`: an instance of a `neural_choiceDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `neural_choiceDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function choice_neural_optimize(model::neural_choiceDDM, options::neural_choice_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false)

    @unpack fit, lb, ub = options
    @unpack θ, data, n, cross = model
    @unpack f = θ

    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -joint_loglikelihood(stack(x,c,fit), model)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)

    model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    converged = Optim.converged(output)

    return model, output

end


"""
"""
θz2(θ::θz) = θz(σ2_i=θ.σ2_i^2, B=θ.B, λ=θ.λ,
        σ2_a=θ.σ2_a^2, σ2_s=θ.σ2_s^2, ϕ=θ.ϕ, τ_ϕ=θ.τ_ϕ)


"""
"""
θ2(θ::θneural_choice) = θneural_choice(θz=θz2(θ.θz), bias=θ.bias, lapse=θ.lapse, θy=θ.θy, f=θ.f)


"""
"""
invθz2(θ::θz) = θz(σ2_i=abs(sqrt(θ.σ2_i)), B=θ.B, λ=θ.λ,
        σ2_a=abs(sqrt(θ.σ2_a)), σ2_s=abs(sqrt(θ.σ2_s)), ϕ=θ.ϕ, τ_ϕ=θ.τ_ϕ)


"""
"""
invθ2(θ::θneural_choice) = θneural_choice(θz=invθz2(θ.θz), bias=θ.bias, lapse=θ.lapse, θy=θ.θy, f=θ.f)


"""
    choice_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function choice_loglikelihood(x::Vector{T}, model::neural_choiceDDM; remap::Bool=false) where {T <: Real}

    @unpack data,θ,n,cross = model
    @unpack f = θ
    if remap
        model = neural_choiceDDM(θ2(θneural_choice(x, f)), data, n, cross)
    else
        model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    end
    choice_loglikelihood(model)

end


"""
    choice_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
choice_loglikelihood(model::neural_choiceDDM) = sum(log.(vcat(choice_likelihood(model)...)))


"""
    choice_likelihood(model)

Arguments: `neural_choiceDDM` instance

Returns: `array` of `array` of P(d|θ, Y)
"""
function choice_likelihood(model::neural_choiceDDM)

    @unpack data,θ,n,cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data ->
            choice_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)

end


"""
"""
function choice_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack choice = data
    @unpack θz, bias, lapse = θ

    P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)[2]
    sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2

end


"""
    joint_loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.

"""
function joint_loglikelihood(x::Vector{T}, model::neural_choiceDDM; remap::Bool=false) where {T <: Real}

    @unpack data,θ,n,cross = model
    @unpack f = θ
    if remap
        model = neural_choiceDDM(θ2(θneural_choice(x, f)), data, n, cross)
    else
        model = neural_choiceDDM(θneural_choice(x, f), data, n, cross)
    end
    joint_loglikelihood(model)

end


"""
    joint_loglikelihood(model)

Given parameters θ and data (inputs and choices) computes the LL for all trials
"""
joint_loglikelihood(model::neural_choiceDDM) = sum(log.(vcat(vcat(joint_likelihood(model)...)...)))


"""
    joint_likelihood(model)

Arguments: `neural_choiceDDM` instance

Returns: `array` of `array` of P(d, Y|θ)
"""
function joint_likelihood(model::neural_choiceDDM)

    @unpack data,θ,n,cross = model
    @unpack θz, θy, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    map((data, θy) -> pmap(data ->
            joint_likelihood(θ,θy,data,P,M,xc,dx,n,cross), data), data, θy)

end


"""
"""
function joint_likelihood(θ, θy, data::neuraldata,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3, n, cross) where {T1,T3 <: Real}

    @unpack choice = data
    @unpack θz, bias, lapse = θ

    c, P = likelihood(θz, θy, data, P, M, xc, dx, n, cross)

    return vcat(c, sum(choice_likelihood!(bias,xc,P,choice,n,dx)) * (1 - lapse) + lapse/2)

end
