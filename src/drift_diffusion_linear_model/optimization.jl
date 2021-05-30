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

    x = Optim.minimizer(output)
    x = stack(x,c,fit)

    model = DDLM(data, options, θDDLM(x))
    converged = Optim.converged(output)

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
function loglikelihood(x::Vector{T1}, data::T2, options::DDLMoptions) where {T1 <: Real, T2<:Vector}
    θ = θDDLM(x)
    options.remap && (θ = θ2(θ))
    sum(map(trialset->loglikelihood(θ, trialset, options), data))
end

"""
    loglikelihood(θ, trialset, options)

Sum the loglikelihood of all trials in a single trial-set

ARGUMENT

-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`
- `options`: specifications of the drift-diffusion linear model

RETURN

-A Float64 indicating the summed log-likelihood of the choice and spike counts given the model parameters

"""
function loglikelihood(θ::θDDLM, trialset::trialsetdata, options::DDLMoptions)

    @unpack σ2_i, B, λ, σ2_a, α, k, bias, lapse = θ
    @unpack a_bases, cross, dt, L2regularizer, n, dt, npostpad_abar = options

    a₀ = pulse_input_DDM.history_influence_on_initial_point(α, k, B, trialset.shifted)
    P,M,xc,dx = pulse_input_DDM.initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used
    xcᵀ = transpose(xc)

    nprepad_abar = size(a_bases[1])[1]-1
    output = map((trial,a₀)->pulse_input_DDM.latent_one_trial(θ, trial, a₀, M, xc, xcᵀ, dx, options, nprepad_abar, npostpad_abar), trialset.trials, a₀)
    P = map(x->x[1], output)
    abar = map(x->x[2], output)

    sum(sum(abar))

    # choicelikelihood = pmap((P, trial)->sum(pulse_input_DDM.choice_likelihood!(bias,xc,P,trial.choice,n,dx)) * (1 - lapse) + lapse/2, P, trialset.trials)
    # LLchoice = sum(log.(choicelikelihood))
    #
    # Xa = vcat(pmap(a->hcat(map(basis->DSP.filt(basis, a)[nprepad_abar+1:end], a_bases)...), abar)...)
    #
    # nLLspike = pmap(unit->pulse_input_DDM.mean_square_error(trialset.Xtiming, unit.Xautoreg, Xa, unit.y, L2regularizer), trialset.units)
    # nLLspike = mean(nLLspike)*(size(trialset.trials)[1]);
    #
    # sum(LLchoice) - sum(nLLspike)
end

"""
    latent_one_trial

Calculate the distribution of the latent at the end of the trial and also return the mean trajectory of the latent during the trial

ARGUMENTS
-θ: an instance of `θDDLM`
-trial: an instance of `trialdata`
-a₀: a(t=0), the value of the latent variable at time equals to zero
-M: A square matrix of length `n` specifying the P{a{t}|a{t-1}} and the discrete approximation to the Fokker-Planck equation
-xc: A vector of length `n` indicating the center of the bins in latent space
-xcᵀ: transpose of xc
-dx: size of the bins in latent size
-n: Number of latent size bins
-cross: Bool indicating whether cross-stream adaptation is implemented
-dt: time bin size, in seconds
-nprepad_abar: number of time bins before the stimulus onset to pad with the value of `abar` at the time of stimulus onset
-npostpad_abar: number of time bins after when the animal is allowed leave the center port to pad with the value of  value of `abar` at the time when the animal is allowed to leave the center port

RETURNS
-P: the distribution of the latent at the end of the trial
-abar: ̅a(t), a vector indicating the mean of the latent variable at each time step
"""
function latent_one_trial(θ::θDDLM, trial::trialdata, a₀::T1, M::Matrix{T1},
                            xc::Vector{T1}, xcᵀ::T2, dx::T1, options::DDLMoptions, nprepad_abar::Int, npostpad_abar::Int) where {T1<:Real, T2<:Any}

    @unpack clickcounts, clicktimes, choice = trial
    @unpack σ2_i, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θ
    @unpack nT, nL, nR = clickcounts
    @unpack L, R = clicktimes

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=options.cross)

    P = P0(σ2_i, a₀, options.n, dx, xc, options.dt)

    #empty transition matrix for time bins with clicks
    F = zeros(T1, options.n, options.n)

    abar = Vector{T1}(undef, nprepad_abar+nT+npostpad_abar)

    @inbounds for t = 1:nT
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,options.n,options.dt)
        abar[nprepad_abar+t] = xcᵀ*P
    end

    abar[1:nprepad_abar] .= abar[nprepad_abar+1]
    abar[nprepad_abar+nT+1:end] .= abar[nprepad_abar+nT]

    return P, abar
end

# """
#     designmatrix
#
# Make a design matrix from the mean trajectory of the latent variable
#
# =INPUT
#
# -abar: ̅a(t), a vector of vectors of floats indicating the mean of the latent variable. Each element of the outer vector corresponds to a trial, and each element of the inner vector corresponds to a time bin
# -a_bases: a vector of vector of floats corresponding to the kernel through which the mean latent trajectory is filtered. Each element of the outer vector corresponds to a basis, and each element of the inner array corresponds to a time bin
#
# =OUTPUT
#
# - a matrix of floats with a number of rows equal to sum of the number of time bins in each trial and a number of columns equal to the number of regressors. The trials are concatenated along the first dimension.
#
# """
# function designmatrix(abar::T, a_bases::T) where {T<:Vector}
#     npad = size(a_bases)[1]-1
#     vcat(pmap(a->hcat(map(basis->filter(basis, a)[npad+1:end], a_bases)...), abar)...)
# end

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
    x = pulse_input_DDM.flatten(θ)
    index = convert(BitArray, map(x->occursin("σ2", string(x)), collect(fieldnames(θDDLM))))
    x[index]=x[index].^2
    θDDLM(x)
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
