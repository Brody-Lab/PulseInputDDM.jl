"""
"""
function train_and_test(data, options::neuraloptions; seed::Int=1, α1s = 10. .^(-6:7))
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    model = map(α1-> optimize([data[train]], options; α1=α1, show_trace=false)[1], α1s)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]]), model)

    return α1s, model, testLL
    
end



"""
"""
function optimize(data, options::neuraloptions;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.)

    @unpack fit, lb, ub, x0, ncells, f, nparams, npolys = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    #ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, ncells, nparams, f, npolys)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, ncells, nparams, f, npolys) -
        α1 * (x[2] - lb[2]).^2)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = unflatten(x, ncells, nparams, f, npolys)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data, ncells, nparams, f, npolys) where {T1 <: Real}

    θ = unflatten(x, ncells, nparams, f, npolys)
    loglikelihood(θ, data)

end


"""
    gradient(model)
"""
function gradient(model::neuralDDM)

    @unpack θ, data = model
    @unpack ncells, nparams, f, npolys = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, ncells, nparams, f, npolys)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
"""
function loglikelihood(θ, data)

    @unpack θz, θμ, θy = θ

    sum(map((θy, θμ, data) -> sum(pmap(data-> loglikelihood(θz, θμ, θy, data), data,
        batch_size=length(data))), θy, θμ, data))

end


"""
"""
function loglikelihood(model::neuralDDM)

    @unpack θ, data = model
    loglikelihood(θ, data)

end


"""
"""
function loglikelihood(θz, θμ, θy, data::neuraldata)

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = loglikelihood(θz,θμ,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function loglikelihood(θz::θz, θμ, θy, input_data::neuralinputs)

    @unpack binned_clicks, dt = input_data
    @unpack nT = binned_clicks

    a = rand(θz,input_data)
    λ = map((θy,θμ)-> θy(a, θμ(1:nT)), θy, θμ)

    return λ, a

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
    p[1] == 0. ? p[1] += 1e-1 : nothing

    return p

end
