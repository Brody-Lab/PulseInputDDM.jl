"""
    optimize(model, options)

Optimize model parameters for a ([`DDLM`](@ref)) using neural and choice data.

ARGUMENT

- `model`: an instance of `DDLM,` a drift-diffusion linear model

RETURN

- `model`: an instance of `DDLM`
- `output`: output of the optimization
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
    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    model = DDLM(data=data, options=options, θ=θDDLM(x, data))

    return model, output
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
function loglikelihood(x::Vector{T1}, data::Vector{T2}, options::DDLMoptions) where {T1<:Real, T2<:trialsetdata}
    nunits_each_trialset = map(trialset->length(trialset.units), data)
    θ = θDDLM(x, data)
    options.remap && (θ = θ2(θ))
    latentspec = latentspecification(options, θ)
    sum(map((coupling, trialset)->loglikelihood(coupling, latentspec, options, θ, trialset), θ.coupling, data))
end

"""
    latentspecification

Create a container of variables used for caculating the dynamics of the latent variable

INPUT

-options: specifications of the drift-diffusion linear model
-θ: parameters of the model

OUTPUT

an instance of `latentspecification`
"""
function latentspecification(options::DDLMoptions, θ::θDDLM)
    @unpack B, k, λ, σ2_a = θ
    @unpack a_bases, cross, dt, n, npostpad_abar = options
    T1 = typeof(σ2_a)
    xc, dx = bins(B, n)
    typedzero = zero(T1);
    M = transition_M(σ2_a*dt, λ, typedzero, dx, xc, n, dt)
    latentspecification(cross=cross, dt=dt, dx=dx, M=M, n=n, nprepad_abar=size(a_bases[1])[1], npostpad_abar=npostpad_abar, xc=xc)
end
"""
    loglikelihood(θ, trialset, latentspec)

Sum the loglikelihood of all trials in a single trial-set

ARGUMENT

-coupling: The degree to which the neural units in this trialset is coordinated with the latent variable
-a_bases: kernels with which to filter the mean of the latent variable
-latentspec: a container of variables specifying the latent space
-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`

RETURN

-The summed log-likelihood of the choice and spike trains given the model parameters. Note that the likelihood of the spike trains of all units in each trial is normalized to be of the same magnitude as that of the choice, between 0 and 1.

"""
function loglikelihood(coupling::Vector{<:Real}, latentspec::latentspecification, options::DDLMoptions, θ::θDDLM, trialset::trialsetdata) where {T1 <:Vector{<:Float64}, T<:Float64}

    @unpack σ2_a, bias, lapse = θ
    @unpack trials, units, nbins_each_trial, Xtiming = trialset
    @unpack dx, n, npostpad_abar, nprepad_abar, xc = latentspec
    @unpack a_bases, autoreg_bases = options

    abar = map(trial->zeros(typeof(σ2_a), nprepad_abar+trial.clickindices.nT+npostpad_abar), trialset.trials)

    P = forwardpass!(abar, latentspec, θ, trialset)
    ℓℓ_choice = sum(log.(map((P, trial)->sum(choice_likelihood!(bias,xc,P,trial.choice,n,dx)), P, trials)))
    Xa = hcat(map(basis->vcat(pmap(abar->DSP.filt(basis, abar)[nprepad_abar+1:end], abar)...), a_bases)...)
    ℓℓ_spike_train = mean(pmap((coupling,unit)->mean(loglikelihood(autoreg_bases, coupling, nbins_each_trial, unit, Xa, Xtiming)), coupling, units))*size(trials)[1]

    ℓℓ_choice + ℓℓ_spike_train
end

"""
    loglikelihood(autoreg_bases, coupling, nbins_each_trial, unit, Xa, Xtiming)

Loglikelihood of the spike trains of one neural unit across all trials of a trialset

INPUT

-autoreg_bases: Kernels with which the spike history is filtered. Each column is a filter, and each row is a time bin. The first row corresponds to the immediately preceding time bin
-coupling: The fraction of trials in which the neural unit is coordinate with the latent variable
-nbins_each_trial: A vector whose each element indicates the number of time bins in each trial
-unit: spike train and regressor associated with a unit
-Xa: the columns of the design matrix corresponding to the filtered output of the mean of the latent variable. It is a N-by-pₐ matrix, where pₐ is the number of filters.
-Xtiming: the columns of the design matrix corresponding to the timing of events in each trial

OUTPUT

-The loglikelihood of the spike trains. A vector of length `∑T(i)`, where `T(i)` is number of time bins in the i-th trial.
"""
function loglikelihood(autoreg_bases::Matrix{Float64}, coupling::T, nbins_each_trial::Vector{Int}, unit::unitdata, Xa::Matrix{T}, Xtiming::Matrix{Float64}) where {T<:Real}
    @unpack L2regularizer, ℓ₀y, Xautoreg, y = unit
    β = leastsquares(unit, Xa, Xtiming)
    ŷ = predict_spike_train(autoreg_bases, β, nbins_each_trial, Xa, Xtiming)
    e = y-ŷ
    σ² = var(e)
    log.(((coupling/sqrt(2π*σ²)).*exp.(-(e.^2)./2σ²) + (1-coupling).*ℓ₀y))
end

"""
    leastsquares(unit, Xa, Xtiming)

Calculate the coefficients of the regressors for the spike train of one neural unit

INPUT

-unit: instance of `unitdata`
-Xa: columns of the design matrix related to the latent
-Xtiming: columns of the design matrix related to the timing of trial events

RETURN

-A vector of coefficients
"""
function leastsquares(unit::unitdata, Xa::Matrix{T1}, Xtiming::Matrix{Float64}) where {T1<:Real}
    @unpack L2regularizer, Xautoreg, y = unit
    X = hcat(Xautoreg, Xtiming, Xa)
    Xᵀ = transpose(X)
    inv(Xᵀ*X+L2regularizer)*Xᵀ*y
end

"""
    predict_spike_train

Calculate the mean of the linear-Gaussian model of the spike in each time bin

INPUT

-autoreg_bases: Kernels with which the spike history is filtered. Each column is a filter, and each row is a time bin. The first row corresponds to the immediately preceding time bin
-β: Coefficients of all regressors. The coefficients for the autoregressive terms are expected to be at the top.
-nbins_each_trial: A vector whose each element indicates the number of time bins in each trial
-Xa: the columns of the design matrix corresponding to the filtered output of the mean of the latent variable. It is a N-by-pₐ matrix, where pₐ is the number of filters.
-Xtiming: the columns of the design matrix corresponding to the timing of events in each trial

"""
function predict_spike_train(autoreg_bases::Matrix{T1}, β::Vector{T2}, nbins_each_trial::Vector{T3}, Xa::Matrix{T1}, Xtiming::Matrix{T1}) where {T1<:Float64, T2<:Real, T3<:Int}
    size_autoreg_bases = size(autoreg_bases)
    n_autoreg_bins = size_autoreg_bases[1]
    n_autoreg_bases = size_autoreg_bases[2]
    w = autoreg_bases * β[1:n_autoreg_bases] # weight of the autoregressive term in each time bin
    y = hcat(Xtiming, Xa) * β[n_autoreg_bases+1:end]
    Y = pad_and_reshape(nbins_each_trial, y)
    @inbounds for i = 2:maximum(nbins_each_trial)
        for j = 1:min(i-1, n_autoreg_bins)
            Y[:,i] += view(Y,:,i-j) .* w[j]
        end
    end
    Yᵀ = transpose(Y)
    Yᵀ[.!isnan.(Yᵀ)]
end

"""
    pad_and_reshape

Pad the spike train of each trial to have the same number of time bins and return the output as a matrix of size number-of-trials by T, where T is the maximum number of time bins across trials

INPUT

-nbins_each_trial: A vector whose each element indicates the number of time bins in each trial
-y: spike count in each time bin concatenated across trials

OUTPUT

-A nan-padded matrix
"""
function pad_and_reshape(nbins_each_trial::Vector{T1}, y::Vector{T2}) where {T1<:Int, T2<:Real}
    ntrials = length(nbins_each_trial);
    Y = fill(NaN, ntrials, maximum(nbins_each_trial))
    k = 0
    @inbounds for i = 1:ntrials
        Y[i, 1:nbins_each_trial[i]] = view(y, (k+1):(k+=nbins_each_trial[i]))
    end
    return Y
end

"""
    forwardpass!

Propagate the distribution of the latent variable from the beginning to the end of each trial, for all trials in a single trialset

ARGUMENT

-abar: mean trajectory of the latent, organized as a vector of vectors. Elements of the outer array correpsond to trials, and those of the inner array corresponds to time bins
-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`
-`options`: specifications of the drift-diffusion linear model

MODIFICATION

-abar: the mean trajectory of the latent, organized in nested vectors. Each element of the outer array corresponds to a trial and each element of the inner array refers to a time bin. The first time bin is centered on the occurence of the stereoclick in a trial.
-F: The transition matrix specifying P(aₜ|aₜ₋₁,θ)

RETURN

-P: A vector of vectors specifying the probability of the latent variable in each bin. Each element of the outer array corresponds to an individual trial
"""
function forwardpass!(abar::Vector{T1}, latentspec::latentspecification, θ::θDDLM, trialset::trialsetdata) where {T1<:Vector{<:Real}}
    @unpack α, B, k = θ
    @unpack lagged, trials = trialset

    a₀ = pulse_input_DDM.history_influence_on_initial_point(α, B, k, lagged)
    pmap((abar, a₀, trial)->forwardpass!(abar, a₀, latentspec, θ, trial), abar, a₀, trials)
end

"""
    forwardpass!

Propagate the distribution of the latent variable from the beginning to the end of of a single trial

ARGUMENT

-abar: mean trajectory of the latent, organized as a vector of vectors. Elements of the outer array correpsond to trials, and those of the inner array corresponds to time bins
-F: An vector of length equal to the number of trials and whose each element is the transition matrix specifying P(aₜ|aₜ₋₁,θ)
-a₀: value of the latent variable at t=0
-dx: size of the bins into which latent space is discretized
-M: P(aₜ|aₜ₋₁, θ, δₜ=0), i.e., a square matrix representing the transition matrix if no clicks occurred in the current time step
-options: specifications of the drift-diffusion linear model. An instance of `DDLMoptions`
-P: A vector specifying the probability of the latent variable in each bin
-θ: an instance of `θDDLM`
-trial: an instance of `trialdata`
-xc: centers of bins in latent space

MODIFICATION

-abar: the mean trajectory of the latent, organized in nested vectors. Each element of the outer array corresponds to a trial and each element of the inner array refers to a time bin. The first time bin is centered on the occurence of the stereoclick in a trial.
-F: The transition matrix specifying P(aₜ|aₜ₋₁,θ)

RETURN
-P: A vector specifying the probability of the latent variable in each bin
"""
function forwardpass!(abar::Vector{T1}, a₀::T1, latentspec::latentspecification, θ::θDDLM, trial::trialdata) where {T1<:Real}
    @unpack clickindices, clicktimes, choice = trial
    @unpack σ2_i, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θ
    @unpack nT, nL, nR = clickindices
    @unpack L, R = clicktimes
    @unpack cross, dt, dx, n, nprepad_abar, xc = latentspec

    typedzero = zero(T1)
    F = fill(typedzero, n, n)
    P = fill(typedzero, n)
    P[ceil(Int,n/2)] = one(T1)
    transition_M!(F, σ2_i, typedzero, a₀, dx, xc, n, dt)
    P = F * P

    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    @inbounds for t = 1:nT
        P = forward_one_step!(F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,latentspec,P)
        abar[nprepad_abar+t] = sum(xc.*P)
    end

    abar[1:nprepad_abar] .= abar[nprepad_abar+1]
    abar[nprepad_abar+nT+1:end] .= abar[nprepad_abar+nT]

    return P
end

"""
    forward_one_step!(F, λ, σ2_a, σ2_s, t, nL, nR, La, Ra, latentspec, P)

Update the transition matrix and the probability distribution of the latent variable

INPUT

-F: A square matrix representing P(aₜ₋₁|aₜ₋₂,θ, Δₜ₋₁) and of length `n`, the number of bins into which latent space is discretized. Δₜ₋₁ represents the clicks occured up to time t-1
-λ: model parameter representing impulsiveness or leakiness
-σ2_a: diffusion noise
-σ2_s: per click noise
-t: time bin
-nL: the index of the time bin of each left click. A vector of length equal to the number of left clicks
-nR: the index of the time bin of each right click. A vector of length equal to the number of right clicks
-La: the adapted click magnitude of left clicks
-Ra: the adapted click magnitude of right clicks
-latentspec: a container of variables specifying the dynamics of the latent variable
    -M: P(aₜ|aₜ₋₁, θ, δₜ=0), i.e., a square matrix representing the transition matrix if no clicks occurred in the current time step
    -dx: size of the bins into which latent space is discretized
    -xc: center of the bins of latent space
    -n: number of bins of latent space
    -dt: size of time bins
-P: A vector representing P(aₜ₋₁|θ), where θ are set of model parameters. It is of length `n`, the number of bins into which latent space is discretized

MODIFICATION
-F: now represents P(aₜ|aₜ₋₁,θ)

RETURN
-P: a vector representing P(aₜ|θ)
"""
function forward_one_step!(F::Array{TT,2}, λ::TT, σ2_a::TT, σ2_s::TT,
        t::Int, nL::Vector{Int}, nR::Vector{Int},
        La::Vector{TT}, Ra::Vector{TT}, latentspec::latentspecification, P::Vector{TT}) where {TT <: Real}

    @unpack dt, dx, M, n, xc = latentspec

    typedzero = zero(TT)
    any(t .== nL) ? sL = sum(La[t .== nL]) : sL = typedzero
    any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = typedzero
    sLR = sL + sR
    if sLR > typedzero
        σ2 = σ2_s*sLR + σ2_a*dt
        transition_M!(F, σ2, λ, sR-sL, dx, xc, n, dt)
        F * P
    else
        M * P
    end
end

"""
    history_influence_on_initial_point(α, k, B, lagged)

Computes the influence of trial history on the initial point for all the trials of a trial-set

ARGUMENT

-α: The magnitude of the correct side of the previous trial on the initial value of the latent variable
-k: The exponential change with which α decays with the number of trial into the past
-B: bound of the accumulation process
-lagged: An instance of `laggeddata`, containing the lagged choices and rewards of all the trials of a trialset

OUTPUT

a₀: initial value of the latent variable of all the trials of a trialset. A vector with length equal to the number of trials
"""
function history_influence_on_initial_point(α::T, B::T, k::T, lagged::laggeddata) where {T <: Real}
    @unpack answer, eˡᵃᵍ⁺¹ = lagged
    a₀ = vec(sum(α.*answer.*eˡᵃᵍ⁺¹.^k, dims=2))
    min.(max.(a₀, -B), B)
end

"""
    θ2(θ)

Square the values of a subset of parameters (σ2_i, σ2_a, σ2_s)

INPUT

-θ: model parameters
-nunits_each_trialset: Number of neuronal units in each trialset
"""
function θ2(θ::θDDLM)
    @unpack α, B, bias, k, λ, lapse, ϕ, σ2_a, σ2_i, σ2_s, τ_ϕ, coupling = θ
    θDDLM(α=α, B=B, bias=bias, k=k, λ=λ, lapse=lapse, ϕ=ϕ, σ2_a=σ2_a^2, σ2_i=σ2_i^2, σ2_s=σ2_s^2, τ_ϕ=τ_ϕ, coupling=coupling)
end
