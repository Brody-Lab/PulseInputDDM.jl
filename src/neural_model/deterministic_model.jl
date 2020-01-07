"""
"""
function optimize(data, options::neuraloptions;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, x0, ncells, f, nparams = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, ncells, nparams, f)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = unflatten(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data, ncells, nparams, f) where {T1 <: Real}

    θ = unflatten(x, ncells, nparams, f)
    loglikelihood(θ, data)

end


"""
    gradient(model)
"""
function gradient(model::neuralDDM)

    @unpack θ, data = model
    @unpack ncells, nparams, f = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
"""
function loglikelihood(θ, data)

    @unpack θz, θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θz, θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(model::neuralDDM)

    @unpack θ, data = model
    loglikelihood(θ, data)

end


"""
"""
function loglikelihood(θz, θy, data::neuraldata)

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = rand(θz,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function initialize_θy(data, f::String)

    ΔLR =  diffLR.(data)
    spikes = group_by_neuron(data)
    #λ0_byneuron = group_by_neuron(data["λ0"], data["ntrials"], data["N"])

    @unpack dt = data[1].input_data
    θy = map(spikes-> compute_p0(vcat(ΔLR...), vcat(spikes...), dt, f), spikes)

end


"""
"""
function compute_p0(ΔLR, spikes, dt, f; nconds::Int=7)

    #conds_bins, = qcut(vcat(ΔLR...),nconds,labels=false,duplicates="drop",retbins=true)
    conds_bins = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), ΔLR)), ΔLR)

    rate = map(i -> (1/dt)*mean(spikes[conds_bins .== i]),1:nconds)

    c = hcat(ones(size(ΔLR, 1)), ΔLR) \ spikes

    if (f == "Sigmoid")
        p = vcat(minimum(rate), maximum(rate)-minimum(rate), c[2], 0.)
    elseif f == "Softplus"
        p = vcat(minimum(rate), (1/dt)*c[2], 0.)
    end

    #added because was getting log problem later, since rate function canot be negative
    p[1] == 0. ? p[1] += eps() : nothing

    return p

end
