abstract type neural_options_noiseless end


"""
"""
@with_kw struct mixed_options_noiseless <: neural_options_noiseless
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    f::Union{Vector{String}, Vector{Vector{String}}}
    fit::Vector{Bool}
    ub::Vector{Float64}
    x0::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
@with_kw struct Sigmoid_options_noiseless <: neural_options_noiseless
    ncells::Vector{Int}
    nparams::Int = 4
    f::String = "Sigmoid"
    fit::Vector{Bool} = vcat(falses(dimz), trues(sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005],
        repeat([-100.,0.,-10.,-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, Inf, 10., Inf, Inf, 1.2,  1.],
        repeat([100.,100.,10.,10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008],
        repeat([10.,10.,1.,0.], sum(ncells)))
end


"""
"""
@with_kw struct Softplus_options_noiseless <: neural_options_noiseless
    ncells::Vector{Int}
    nparams::Int = 1
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(falses(dimz), trues(sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -10., 0.,   0.,  0., 0.005],
        repeat([-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, 200., 10., Inf, Inf, 1.2,  1.],
        repeat([10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008],
        repeat([1.], sum(ncells)))
end


"""
"""
@with_kw struct θneural_noiseless_mixed{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    f::Union{Vector{String}, Vector{Vector{String}}}
end


"""
"""
@with_kw struct θneural_noiseless{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    ncells::Vector{Int}
    nparams::Int
    f::String
end


#=
"""
"""
@with_kw struct Softplus_options_noiseless <: neural_options_noiseless
    ncells::Vector{Int}
    nparams::Int = 3
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(falses(dimz), trues(sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -5., 0.,   0.,  0.01, 0.005],
        repeat([-100.,-10.,-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, Inf, 10., Inf, Inf, 1.2,  1.],
        repeat([100.,10.,10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0., 15., 0. - eps(), 0., 0., 1.0 - eps(), 0.008],
        repeat([10.,1.,0.], sum(ncells)))
end
=#



"""
"""
@with_kw struct null_options
    ncells::Vector{Int}
    nparams::Int = 4
    f::String = "Sigmoid"
    fit::Vector{Bool} = repeat([true,false,false,false], sum(ncells))
    lb::Vector{Float64} = repeat([-100.,0.,-10.,-10.], sum(ncells))
    ub::Vector{Float64} = repeat([100.,100.,10.,10.], sum(ncells))
    x0::Vector{Float64} = repeat([1e-1,eps(),0.,0.], sum(ncells))
end


"""
"""
@with_kw struct θneural_null{T1} <: DDMθ
    θy::T1
    ncells::Vector{Int}
    nparams::Int
    f::String
end


"""
"""
function optimize(data, options::null_options;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=false,
        outer_iterations::Int=Int(1e1))

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural_null(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, θ)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_null(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
"""
function θneural_null(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[1:nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
        
    θneural_null(θy, ncells, nparams, f)

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural_null)

    @unpack θy = θ
    vcat(collect.(Flatten.flatten.(vcat(θy...)))...)

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data::Union{Vector{Vector{T2}}, Vector{Any}}, 
        θ::θneural_null) where {T1 <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_null(x, ncells, nparams, f)
    loglikelihood(θ, data)

end



"""
"""
function loglikelihood(θ::θneural_null, 
        data::Union{Vector{Vector{T1}}, Vector{Any}}) where T1 <: neuraldata

    @unpack θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(θy::Vector{T1}, data::neuraldata) where T1 <: DDMf

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ = loglikelihood(θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function loglikelihood(θy::Vector{T1}, input_data::neuralinputs) where T1 <: DDMf

    @unpack λ0, dt = input_data

    λ = map((θy,λ0)-> θy(zeros(length(λ0)), λ0), θy, λ0)

end


"""
"""
function train_and_test(data, options::T1; seed::Int=1, α1s = 10. .^(-6:7)) where T1 <: neural_options_noiseless
    
    ntrials = length(data)
    train = sample(Random.seed!(seed), 1:ntrials, ceil(Int, 0.9 * ntrials), replace=false)
    test = setdiff(1:ntrials, train)
      
    model = map(α1-> optimize([data[train]], options; α1=α1, show_trace=false)[1], α1s)   
    testLL = map(model-> loglikelihood(model.θ, [data[test]]), model)

    return α1s, model, testLL
    
end


function sigmoid_prior(x::Vector{T1}, data::Union{Vector{Vector{T2}}, Vector{Any}}, 
        θ::Union{θneural_noiseless, θneural_noiseless_mixed, θneural, θneural_mixed}; 
        sig_σ::Float64=1.) where {T1 <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_noiseless(x, ncells, nparams, f)
    
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
function optimize(data, options::T1;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), α1::Float64=0.,
        sig_σ::Float64=1.) where T1 <: neural_options_noiseless

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural_noiseless(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, θ) + sigmoid_prior(stack(x,c,fit), data, θ; sig_σ=sig_σ))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_noiseless(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::Union{θneural_noiseless, θneural_noiseless_mixed})

    @unpack θy, θz = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, 
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    loglikelihood(x, data, ncells)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T1}, data::Union{Vector{Vector{T2}}, Vector{Any}}, 
        θ::Union{θneural_noiseless, θneural_noiseless_mixed}) where {T1 <: Real, T2 <: neuraldata}

    @unpack ncells, nparams, f = θ
    θ = θneural_noiseless(x, ncells, nparams, f)
    loglikelihood(θ, data)

end


"""
"""
function θneural_noiseless(x::Vector{T}, ncells::Vector{Int}, nparams::Vector{Vector{Int}}, 
        f::Vector{Vector{String}}) where {T <: Real}
    
    #borg = vcat(dimz, dimz.+cumsum(nparams .* ncells))
    #borg = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    #borg = partition.(borg, nparams);
            
    #θy = map((x,f)-> map(x-> f(x...), x), borg, getfield.(Ref(@__MODULE__), Symbol.(f)))
    
    borg = vcat(dimz,dimz.+cumsum(vcat(nparams...)))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_noiseless_mixed(θz(x[1:dimz]...), θy, ncells, nparams, f)

end


"""
"""
function θneural_noiseless(x::Vector{T}, ncells::Vector{Int}, nparams::Vector{Int}, f::Vector{String}) where {T <: Real}
    
    borg = vcat(dimz, dimz.+cumsum(nparams .* ncells))
    borg = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    borg = partition.(borg, nparams);
            
    θy = map((x,f)-> map(x-> f(x...), x), borg, getfield.(Ref(@__MODULE__), Symbol.(f)))
    
    θneural_noiseless_mixed(θz(x[1:dimz]...), θy, ncells, nparams, f)

end


"""
"""
function θneural_noiseless(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[dimz + 1:dimz + nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
        
    θneural_noiseless(θz(Tuple(x[1:dimz])...), θy, ncells, nparams, f)

end



"""
    gradient(model)
"""
function gradient(model::neuralDDM)

    @unpack θ, data = model
    x = flatten(θ)
    ℓℓ(x) = -loglikelihood(x, data, θ)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
"""
function loglikelihood(θ::Union{θneural_noiseless, θneural_noiseless_mixed}, 
        data::Union{Vector{Vector{T1}}, Vector{Any}}) where T1 <: neuraldata

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
    map((spikes,f)-> θy(vcat(ΔLR...), vcat(spikes...), dt, f), spikes,f)

end


"""
"""
function θy(data, f::String)

    ΔLR = diffLR.(data)
    spikes = group_by_neuron(data)

    @unpack dt = data[1].input_data
    map(spikes-> θy(vcat(ΔLR...), vcat(spikes...), dt, f), spikes)

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