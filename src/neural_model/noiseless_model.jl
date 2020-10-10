"""
"""
@with_kw struct noiseless_neuralDDM{T,U} <: DDM
    θ::T
    data::U
end


"""
"""
@with_kw struct neural_options_noiseless
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
function neural_options_noiseless(f)
    
    nparams, ncells = nθparams(f)
    fit = vcat(falses(dimz), trues.(nparams)...)
        
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

    neural_options_noiseless(fit=fit, ub=ub, lb=lb)
    
end


"""
"""
@with_kw struct θneural_noiseless{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end



"""
"""
function θneural_noiseless(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(dimz,dimz.+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_noiseless(θz(x[1:dimz]...), θy, f)

end


"""
"""
function train_and_test(data, x0, options::T1; seed::Int=1, α1s = 10. .^(-6:7)) where T1 <: neural_options_noiseless
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    model = map(α1-> optimize([data[train]], x0, options; α1=α1, show_trace=false)[1], α1s)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]]), model)

    return α1s, model, testLL
    
end


function sigmoid_prior(x::Vector{T1}, θ::Union{θneural_noiseless, θneural}; 
        sig_σ::Float64=1.) where {T1 <: Real, T2 <: neuraldata}

    @unpack f = θ
    θ = θneural_noiseless(x, f)
    
    if typeof(f) == String
        if f == "Sigmoid"
            sum(map(x-> sum(logpdf.(Normal(0., sig_σ), map(x-> x.c, x))), θ.θy))
        else
            0.
        end
    else    
        sum(map(x-> sum(logpdf.(Normal(0., sig_σ), x.c)), vcat(θ.θy...)[vcat(f...) .== "Sigmoid"]))
    end
    
end


"""
"""
function optimize(data, x0_y, f, options::neural_options_noiseless;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.,
        sig_σ::Float64=1., x0_z::Vector{Float64}=[0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008])

    @unpack fit, lb, ub = options
    
    x0 = vcat(x0_z, x0_y) 
    θ = θneural_noiseless(x0, f)
    model = noiseless_neuralDDM(θ, data)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), model) + sigmoid_prior(stack(x,c,fit), θ; sig_σ=sig_σ))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = noiseless_neuralDDM(θneural_noiseless(x, f), data)
    converged = Optim.converged(output)

    return model, output

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural_noiseless)

    @unpack θy, θz = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    gradient(model)
"""
function gradient(model::noiseless_neuralDDM)

    @unpack θ = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, model)

    ForwardDiff.gradient(ℓℓ, x)

end



"""
    loglikelihood(x, model)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, model::noiseless_neuralDDM) where {T1 <: Real}
    
    @unpack data,θ = model
    @unpack f = θ 
    model = noiseless_neuralDDM(θneural_noiseless(x, f), data)
    loglikelihood(model)

end


"""
"""
function loglikelihood(model::noiseless_neuralDDM)

    @unpack θ, data = model
    @unpack θz, θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θz, θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(θz::θz, θy::Vector{T1}, data::neuraldata) where T1 <: DDMf

    @unpack spikes, input_data = data
    @unpack λ0, dt = input_data
    
    ΔLR = diffLR(data)
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
