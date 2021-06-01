"""
    optimize(model, options)

Optimize model parameters for a ([`DDLM`](@ref)) using neural and choice data.

ARGUMENT

- `model`: an instance of `DDLM,` a drift-diffusion linear model

RETURN

- `model`: an instance of `DDLM`

"""
function optimize(model::DDLM;
        x_tol::Float64=1e-10, f_tol::Float64=1e-9, g_tol::Float64=1e-3,
        iterations::Int=Int(1e2), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        scaled::Bool=false, extended_trace::Bool=false)

    @unpack θ, data, options = model
    @unpack fit, lb, ub = options

    x0 = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, options)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, scaled=scaled,
        extended_trace=extended_trace)
    Optim.converged(output) || error("Failed to converge in $(Optim.iterations(output)) iterations")
    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    DDLM(data=data, options=options, θ=θDDLM(x))
end

"""
    loglikelihood(x, data, options)

A wrapper function that computes the loglikelihood of a drift-diffusion linear model given the model parameters, data, and model options. Used
in optimization, Hessian and gradient computation.

ARGUMENT

- `x`: a vector of model parameters
- 'data': a vector of `trialsetdata`
- `options`: specifications of the drift-diffusion linear model

Returns:

- The loglikelihood of choices and spikes counts given the model parameters, pulse timing, trial history, and model specifications, summed across trials and trial-sets
"""
function loglikelihood(x::Vector{T1}, data::T2, options::DDLMoptions) where {T1 <: Real, T2<:Vector}
    θ = θDDLM(x)
    options.remap && (θ = θ2(θ))
    sum(map(trialset->loglikelihood(θ, trialset, options), data))
end

"""
    loglikelihood(θ, trialset, options, M, xc, dx)

Sum the loglikelihood of all trials in a single trial-set

ARGUMENT

-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`
-`options`: specifications of the drift-diffusion linear model

RETURN

-A Float64 indicating the summed log-likelihood of the choice and spike counts given the model parameters

"""
function loglikelihood(θ::θDDLM, trialset::trialsetdata, options::DDLMoptions)

    @unpack a_bases = options

    abar, choicelikelihood = mean_latent_choice_likelihood(θ::θDDLM, trialset::trialsetdata, options::DDLMoptions)

    nprepad_abar = size(a_bases[1])[1]-1
    Xa = vcat(pmap(a->hcat(map(basis->DSP.filt(basis, a)[nprepad_abar+1:end], a_bases)...), abar)...)

    nLLspike = pmap(unit->pulse_input_DDM.mean_square_error(trialset.Xtiming, unit.Xautoreg, Xa, unit.y, L2regularizer), trialset.units)
    nLLspike = mean(nLLspike)*(size(trialset.trials)[1]);

    sum(log.(choicelikelihood)) - sum(nLLspike)
end

"""
    mean_latent_choice_likelihood

Calculate the mean trajectory of the latent and the likelihood of the choice for every trial in a trialset

ARGUMENT

-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`
-`options`: specifications of the drift-diffusion linear model

RETURN

-abar: the mean trajectory of the latent, organized in nested vectors. Each element of the outer array corresponds to a trial and each element of the inner array refers to a time bin. The first time bin is centered on the occurence of the stereoclick in a trial.
-choicelikelihood: the likelihood of the observed choice in each trial. A vector.
"""
function mean_latent_choice_likelihood(θ::θDDLM, trialset::trialsetdata, options::DDLMoptions)
    @unpack α, B, bias, k, λ, lapse, σ2_a, σ2_i= θ
    @unpack a_bases, cross, dt, L2regularizer, n, npostpad_abar = options

    xc, dx = bins(B, n)
    M = transition_M(σ2_a*dt, λ, zero(typeof(σ2_a)), dx, xc, n, dt)
    a₀ = history_influence_on_initial_point(α, k, B, trialset.shifted)

    nprepad_abar = size(a_bases[1])[1]-1
    output = pmap((a₀, trial)->mean_latent_choice_likelihood(a₀, cross, dt, dx, M, n, npostpad_abar, nprepad_abar, θ, trial, xc), a₀, trialset.trials)
    abar = map(x->x[1], output)
    choicelikelihood = map(x->x[2], output)
    return abar, choicelikelihood
end

"""
    mean_latent_choice_likelihood

Calculate the mean trajectory of the latent and the likelihood of the choice for a single trial

ARGUMENTS
-θ: an instance of `θDDLM`
-trial: an instance of `trialdata`
-a₀: a(t=0), the value of the latent variable at time equals to zero
-M: A square matrix of length `n` specifying the P{a{t}|a{t-1}} and the discrete approximation to the Fokker-Planck equation
-xc: A vector of length `n` indicating the center of the bins in latent space
-dx: size of the bins in latent size
-n: Number of latent size bins
-cross: Bool indicating whether cross-stream adaptation is implemented
-dt: time bin size, in seconds

RETURNS
-P: the distribution of the latent at the end of the trial
-abar: ̅a(t), a vector indicating the mean of the latent variable at each time step
"""
function mean_latent_choice_likelihood(a₀::T1, cross::Bool, dt::Float64, dx::T2,
    M::Matrix{T1}, n::Int, θ::θDDLM, trial::trialdata, xc::Vector{T1}) where {T1, T2<:Real}

    @unpack clickcounts, clicktimes, choice = trial
    @unpack σ2_i, λ, lapse, σ2_a, σ2_s, ϕ, τ_ϕ = θ
    @unpack nT, nL, nR = clickcounts
    @unpack L, R = clicktimes

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    P = P0(a₀, dt, dx, lapse, n, σ2_i)

    #empty transition matrix for time bins with clicks
    F = zeros(T1, n, n)

    abar = Vector{T1}(undef, nprepad_abar+nT+npostpad_abar)
    xcᵀ = transpose(xc)

    @inbounds for t = 1:nT
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        abar[t] = xcᵀ*P
    end
    abar[1:nprepad_abar] .= abar[nprepad_abar+1]
    abar[nprepad_abar+nT+1:end] .= abar[nprepad_abar+nT]

    choicelikelihood = sum(choice_likelihood!(bias,xc,P,choice,n,dx))

    return abar, choicelikelihood
end

"""
    mean_square_error

Predict the spike counts of a neural unit using a multilinear regression model and return the in sample mean-square-error

ARGUMENT

-Xtiming: columns of the design matrix representing regressors that depend on the timing of events in each trial
-Xautoreg: columns of the design matrix representing autoregressive terms that depend on spiking history
-Xa: regressors that depend on the mean of the latent variable
-L2regularizer: penalty matrix for L2 regularization
"""
function mean_square_error(Xtiming::Matrix{T}, Xautoreg::Matrix{T}, Xa::T2, y::Vector{T}, L2regularizer::Matrix{T}) where {T<:Real, T2<:Any}
    X = hcat(Xtiming, Xautoreg, Xa)
    mean((X*(inv(transpose(X)*X+L2regularizer)*transpose(X)*y)-y).^2)
end

"""
    θ2(θ)

Square the values of a subset of parameters (σ2_i, σ2_a, σ2_s)
"""
function θ2(θ::θDDLM)
    @unpack θz, θh, bias, lapse = θ
    x = pulse_input_DDM.flatten(θz)
    index = convert(BitArray, map(x->occursin("σ2", string(x)), collect(fieldnames(θz))))
    x[index]=x[index].^2
    θDDLM(θz=θz(x), θh=θh, bias=bias, lapse=lapse)
end

"""
    history_influence_on_initial_point(α, k, B, shifted)

Computes the influence of trial history on the initial point for all the trials of a trial-set

Arguments:

-`θhist`: trial history parameters (['θh'](@ref))
-'B': bound
-'shifted': (['trialshifted'](@ref))
"""
function history_influence_on_initial_point(α::T, k::T, B::T, shifted::trialshifted) where {T <: Real}
    @unpack choice, reward, shift = shifted
    a₀ = sum(α.*choice.*reward.*exp.(k.*(shift.+1)), dims=2)[:,1]
    min.(max.(a₀, -B), B)
end

"""
    P0(a₀, dt, dx, lapse, n, σ2_i)

The initial distribution of the latent variable

ARGUMENT

-a₀: the latent variable at the beginning of the trial
-dt: the size of the time bin, in seconds
-dx: the size of each bin of the latent variable
-lapse: the fraction of the trials the subject begins (and remains) at one of the two bounds, chosen randomly
-n: number of bins the latent variable is divided into
-σ2_i: the variance of the initial value
"""
function P0(a₀::T1, dt::Float64, dx::T2, lapse::T1, n::Int, σ2_i::T1) where {T1,T2 <: Any}
    P = zeros(T1,n)
    P[ceil(Int,n/2)] = one(T1) - lapse
    P[1], P[n] = lapse/2., lapse/2.
    M = transition_M(σ2_i, zero(T1), a₀, dx, xc, n, dt)
    P = M * P
end
