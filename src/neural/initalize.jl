"""
    initalize(data, f)

Returns: initializition of neural parameters. Module-defined class `θy`.

"""
function initalize(data, f::Vector{Vector{String}})
    
    θy0 = θy.(data, f) 
    x0 = vcat([0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008], vcat(vcat(θy0...)...)) 
    θ = θneural(x0, f)
    fitbool, lb, ub = neural_options_noiseless(f)
    model0 = noiseless_neuralDDM(θ=θ, fit=fitbool, lb=lb, ub=ub)  
    model0, = fit(model0, data; iterations=10, outer_iterations=1)
    
    return model0
    
end


"""
    fit(model, options)

fit model parameters for a `noiseless_neuralDDM`.

Arguments: 

- `model`: an instance of a `noiseless_neuralDDM`.
- `options`: some details related to the optimzation, such as which parameters were fit (`fit`), and the upper (`ub`) and lower (`lb`) bounds of those parameters.

Returns:

- `model`: an instance of a `noiseless_neuralDDM`.
- `output`: results from [`Optim.optimize`](@ref).

"""
function fit(model::noiseless_neuralDDM, data;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=false,
        outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, θ = model
    @unpack f = θ
    
    x0 = PulseInputDDM.flatten(θ)      
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -loglikelihood(stack(x,c,fit), model, data)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model.θ = θneural(x, f)

    return model, output

end



"""
    loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, model::noiseless_neuralDDM, data) where {T1 <: Real}
    
    @unpack θ,fit,lb,ub = model
    @unpack f = θ 
    model = noiseless_neuralDDM(θ=θneural(x, f),fit=fit,lb=lb,ub=ub)
    loglikelihood(model, data)

end


"""
"""
function loglikelihood(model::noiseless_neuralDDM, data)

    @unpack θ = model
    @unpack θz, θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θz, θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(θz::θz, θy::Vector{T1}, data::neuraldata) where T1 <: DDMf

    @unpack spikes, input_data = data
    @unpack λ0, dt = input_data
    
    #ΔLR = diffLR(data)
    ΔLR = rand(θz, input_data)
    #λ = loglikelihood(θz,θy,λ0,ΔLR)
    λ = map((θy,λ0)-> θy(ΔLR, λ0), θy, λ0)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


#=
"""
"""
function loglikelihood(θz::θz, θy::Vector{T1}, λ0, ΔLR) where T1 <: DDMf

    #@unpack λ0, dt = input_data

    #a = rand(θz,input_data)
    λ = map((θy,λ0)-> θy(ΔLR, λ0), θy, λ0)

    #return λ, a

end
=#

"""
"""
function θy(data, f::Vector{String})

    ΔLR = diffLR.(data)
    spikes = group_by_neuron(data)

    @unpack dt = data[1].input_data
    map((spikes,f)-> θy(vcat(ΔLR...), vcat(spikes...), dt, f), spikes, f)

end


"""
"""
function θy(ΔLR, spikes, dt, f; nconds::Int=7)

    conds_bins = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), ΔLR)), ΔLR)

    rate = map(i -> (1/dt)*mean(spikes[conds_bins .== i]), 1:nconds)

    c = hcat(ones(size(ΔLR, 1)), ΔLR) \ spikes

    if (f == "Sigmoid")
         #p = vcat(minimum(rate) - mean(rate), maximum(rate)- mean(rate), c[2])
        p = vcat(minimum(rate) - mean(rate), maximum(rate)- mean(rate), c[2], 0.)
        #p = vcat(minimum(rate), maximum(rate)- minimum(rate), c[2], 0.)
    elseif f == "Softplus"
        #p = vcat(minimum(rate) - mean(rate), (1/dt)*c[2], 0.)
        #p = vcat(eps(), (1/dt)*c[2], 0.)
        #p = vcat(minimum(rate) - mean(rate), (1/dt)*c[2])
        #p = vcat((1/dt)*c[2], 0.)
        p = vcat((1/dt)*c[2])
    end

    #added because was getting log problem later, since rate function canot be negative
    p[1] == 0. ? p[1] += 1e-1 : nothing

    return p

end
