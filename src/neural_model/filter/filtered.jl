"""
"""
@with_kw struct filtinputs{T1,T2,T3}
    clicks::T1
    binned_clicks::T2
    λ0::Vector{Vector{Float64}}
    LR::T3
    dt::Float64
    centered::Bool
    delay::Int
    pad::Int
end


"""
"""
@with_kw struct filtdata <: DDMdata
    input_data::filtinputs
    spikes::Vector{Vector{Int}}
    ncells::Int
    choice::Bool
end


"""
"""
@with_kw struct filtoptions <: neural_options
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    filt_len::Int = 50
    shift::Int=0
    f::Union{Vector{String}, Vector{Vector{String}}}
    fit::Vector{Bool}
    ub::Vector{Float64}
    x0::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
@with_kw struct θneural_filt{T1, T2} <: DDMθ
    w::Vector{T1}
    θy::T2
    ncells::Vector{Int}
    nparams::Union{Vector{Int}, Vector{Vector{Int}}}
    f::Union{Vector{String}, Vector{Vector{String}}}
    filt_len::Int
end


"""
"""
function make_filt_data(data, filt_len; shift=0)
    
    @unpack input_data, spikes, ncells, choice = data
    @unpack binned_clicks, clicks, dt, centered, λ0, delay, pad = input_data

    L,R = binLR(binned_clicks, clicks, dt)
    LR = -L + R
    #LRX = map(i-> vcat(missings(Int, max(0, filt_len - i)), LR[max(1,i-filt_len+1):i]), 1:length(LR))
    LRX = map(i-> vcat(missings(Int, max(0, filt_len - (i+shift))), 
            LR[max(1,(i+shift)-filt_len+1): min(length(LR), i+shift)], 
            missings(Int, max(0, -(length(LR) - (i+shift))))), 1:length(LR))

    filtdata(filtinputs(clicks, binned_clicks, λ0, LRX, dt, centered, delay, pad), spikes, ncells, choice)
    
end


function prior(x::Vector{T1}, filt_len; sig_σ::Float64=1.) where {T1 <: Real}
    
    - sig_σ * sum(diff(x[1:filt_len]).^2)
    
end


"""
"""
function optimize(data, options::filtoptions;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, sig_σ::Float64=1.)

    @unpack fit, lb, ub, x0, ncells, f, nparams, filt_len, shift = options
    
    filt_data = map(data-> make_filt_data.(data, Ref(filt_len); shift=shift), data)
    θ = θneural_filt(x0, ncells, nparams, f, filt_len)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), filt_data, θ) + prior(stack(x,c,fit), filt_len; sig_σ=sig_σ))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_filt(x, ncells, nparams, f, filt_len)
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
function loglikelihood(x::Vector{T1}, data, θ::θneural_filt) where {T1 <: Real}

    @unpack ncells, nparams, f, filt_len = θ
    θ = θneural_filt(x, ncells, nparams, f, filt_len)
    loglikelihood(θ, data)

end


"""
"""
function θneural_filt(x::Vector{T}, ncells::Vector{Int}, nparams::Vector{Vector{Int}}, 
        f::Vector{Vector{String}}, filt_len::Int) where {T <: Real}
    
    borg = vcat(filt_len, filt_len.+cumsum(vcat(nparams...)))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_filt(x[1:filt_len], θy, ncells, nparams, f, filt_len)

end


"""
"""
function flatten(θ::θneural_filt)

    @unpack w, θy = θ
    vcat(w, vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
"""
function loglikelihood(θ::θneural_filt, data)

    @unpack w, θy = θ

    sum(map((θy, data) -> sum(loglikelihood.(Ref(w), Ref(θy), data)), θy, data))

end


"""
"""
function loglikelihood(w, θy, data::filtdata)

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = loglikelihood(w,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end



"""
"""
function loglikelihood(w, θy, input_data::filtinputs)

    @unpack binned_clicks, λ0, dt = input_data
    @unpack nT = binned_clicks

    a = rand(w, input_data)
    λ = map((θy, λ0)-> θy(a, λ0), θy, λ0)

    return λ, a

end


"""
"""
function rand(w, inputs)

    @unpack LR = inputs
    
    afilt.(Ref(w), LR)      
    
end


"""
"""
afilt(w, LR) = sum(skipmissing(w .* LR))


#=


"""
    Sample rates from latent model with multiple rngs, to average over
"""
function synthetic_λ(θ::θfilt, data; nconds::Int=2)

    @unpack θτ,θμ,θy,ncells = θ

    μ_λ = rand.(Ref(θτ), θμ, θy, data)
    
    μ_c_λ = cond_mean.(μ_λ, data, ncells; nconds=nconds)
    
    return μ_λ, μ_c_λ

end


"""
    Sample all trials over one session
"""
function rand(θτ::θτ, θμ, θy, data)
    
    pmap(data -> loglikelihood(θτ,θμ,θy,data.input_data)[1], data)

end

=#