abstract type neural_choice_options end


"""
"""
@with_kw struct θneural_choice{T1, T2, T3} <: DDMθ
    θz::T1
    bias::T2
    θy::T3
    ncells::Vector{Int}
    nparams::Int
    f::String
end


"""
"""
@with_kw struct neural_choice_data <: DDMdata
    input_data::neuralinputs
    choice::Bool
    spikes::Vector{Vector{Int}}
    ncells::Int
end

"""
"""
neural_choice_data(input_data, choices, spikes::Vector{Vector{Vector{Int}}}, ncells::Int) =  
    neural_choice_data.(input_data,choices,spikes,ncells)


"""
"""
@with_kw struct Softplus_choice_options <: neural_choice_options
    ncells::Vector{Int}
    nparams::Int = 1
    f::String = "Softplus"
    fit::Vector{Bool} = vcat(trues(dimz + 1 + sum(ncells)*nparams))
    lb::Vector{Float64} = vcat([0., 8.,  -10., 0.,   0.,  0., 0.005],
        [-30], repeat([-10.], sum(ncells)))
    ub::Vector{Float64} = vcat([Inf, 200., 10., Inf, Inf, 1.2,  1.],
        [30], repeat([10.], sum(ncells)))
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008],
        [0.], repeat([1.], sum(ncells)))
end


"""
"""
function θneural_choice(x::Vector{T}, ncells::Vector{Int}, nparams::Int, f::String) where {T <: Real}
    
    dims2 = vcat(0,cumsum(ncells))

    blah = Tuple.(collect(partition(x[dimz + 1 + 1:dimz + 1 + nparams*sum(ncells)], nparams)))
    
    if f == "Sigmoid"
        blah2 = map(x-> Sigmoid(x...), blah)
    elseif f == "Softplus"
        blah2 = map(x-> Softplus(x...), blah)
    end
    
    θy = map(idx-> blah2[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1]) 
    bias = x[dimz+1]
        
    θneural_choice(θz(Tuple(x[1:dimz])...), bias, θy, ncells, nparams, f)

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural_choice)

    @unpack θy, θz, bias = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end


"""
    optimize_model(pz, py, data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

BACK IN THE DAY, TOLS USED TO BE x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize(data, options::T1; n::Int=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(1e1), scaled::Bool=false,
        extended_trace::Bool=false, σ::Vector{Float64}=[0.], 
        μ::Vector{Float64}=[0.], do_prior::Bool=false) where T1 <: neural_choice_options

    @unpack fit, lb, ub, x0, ncells, f, nparams = options
    
    θ = θneural_choice(x0, ncells, nparams, f)

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)
    ℓℓ(x) = -(loglikelihood(stack(x,c,fit), data, θ; n=n) + Float64(do_prior) * logprior(stack(x,c,fit)[1:dimz],μ,σ))

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = θneural_choice(x, ncells, nparams, f)
    model = neuralDDM(θ, data)
    converged = Optim.converged(output)

    return model, output

end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, data::Vector{Vector{T2}}, θ::θneural_choice; n::Int=53) where {T <: Real, T2 <: neural_choice_data}

    @unpack ncells, nparams, f = θ
    θ = θneural_choice(x, ncells, nparams, f)
    loglikelihood(θ, data; n=n)

end


"""
    LL_all_trials(pz, py, data; n=53)

Computes the log likelihood for a set of trials consistent with the observed neural activity on each trial.
"""
function loglikelihood(θ::θneural_choice, data::Vector{Vector{T1}}; n::Int=53) where {T1 <: neural_choice_data}

    @unpack θz, θy, bias = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack dt = data[1][1].input_data

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt)

    sum(map((data, θy) -> sum(pmap(data -> 
                    loglikelihood(θz,bias,θy,data, P, M, xc, dx; n=n), data)), data, θy))

end


"""
"""
function loglikelihood(θz,bias,θy,data::neural_choice_data,
        P::Vector{T1}, M::Array{T1,2},
        xc::Vector{T1}, dx::T3; n::Int=53) where {T1,T3 <: Real}

    @unpack λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack spikes, input_data, choice = data
    @unpack binned_clicks, clicks, dt, λ0, centered, delay, pad = input_data
    @unpack nT, nL, nR = binned_clicks
    @unpack L, R = clicks

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R)

    F = zeros(T1,n,n) #empty transition matrix for time bins with clicks
    
    time_bin = (-(pad-1):nT+pad) .- delay
    
    c = Vector{T1}(undef, length(time_bin))

    @inbounds for t = 1:length(time_bin)

        if time_bin[t] >= 1
            P, F = latent_one_step!(P, F, λ, σ2_a, σ2_s, time_bin[t], nL, nR, La, Ra, M, dx, xc, n, dt)
        end

        #weird that this wasn't working....
        #P .*= vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
        #                        k[t]), spikes, θy, λ0))), xc)...)
        
        P = P .* (vcat(map(xc-> exp(sum(map((k,θy,λ0)-> logpdf(Poisson(θy(xc,λ0[t]) * dt),
                        k[t]), spikes, θy, λ0))), xc)...))
        
        c[t] = sum(P)
        P /= c[t]

    end
    
    return sum(log.(c)) + log(sum(choice_likelihood!(bias,xc,P,choice,n,dx)))

end