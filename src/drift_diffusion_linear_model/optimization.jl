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

    x0 = vec(θ)
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
function loglikelihood(x::Vector{T1}, data::Vector{trialsetdata}, options::DDLMoptions) where {T1 <: AbstractFloat}
    θ = θDDLM(x)
    options.remap && (θ = θ2(θ))
    @unpack σ2_i, B, λ, σ2_a = θ.θz
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
function loglikelihood(θ::θDDLM, trialset::trialsetdata, options::DDLMoptions) where {T <: AbstractFloat}

    @unpack θz, θh, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack a_bases, cross, dt, L2regularizer, n = options

    a₀ = history_influence_on_initial_point(θh, B, trialset.shifted)
    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used

    nprepad_abar = size(a_bases)[1]-1
    P, abar = pmap((trial,a₀)->latent_one_trial(θz, trial, a₀, M, transpose(xc), dx, n, cross, nprepad_abar), trialset.trials, a₀)

    choicelikelihood = pmap((P, trial)->sum(choice_likelihood!(bias,xc,P,trial.choice,n,dx)) * (1 - lapse) + lapse/2, P, trialset.trials)
    LLchoice = sum(log.(choicelikelihood))

    Xa = designmatrix(abar, a_bases)
    nLLspike = pmap(unit->mean_square_error(trialset.Xtiming, unit.Xautoreg, Xa, unit.y, L2regularizer), trialset.units)
    nLLspike = mean(nLLspike)*size(trialset.trials)[1];

    sum(LLchoice) - sum(nLLspike)
end

"""
    latent_one_trial

Calculate the distribution of the latent at the end of the trial and also return the mean trajectory of the latent during the trial

ARGUMENTS
-θ: an instance of `θz`
-trial: an instance of `trialdata`
-a₀: a(t=0), the value of the latent variable at time equals to zero
-M: A square matrix of length `n` specifying the P{a{t}|a{t-1}} and the discrete approximation to the Fokker-Planck equation
-xcᵀ: A row vector of length `n` indicating the center of the bins in latent space
-dx: size of the bins in latent size
-n: Number of latent size bins
-cross: Bool indicating whether cross-stream adaptation is implemented
-nprepad_abar: number of time bins before the stimulus onset to pad to at the beginning of the `abar` with the value of `abar` at the time of stimulus onset

RETURNS
-P: the distribution of the latent at the end of the trial
-abar: ̅a(t), a vector indicating the mean of the latent variable at each time step
"""
function latent_one_trial(θ::θz, trial::trialdata, a₀::TT, M::Matrix{TT},
                            xcᵀ::Matrix{TT}, dx::TT, n::Int, cross::Bool, nprepad_abar::Int) where {TT <: Real}

    @unpack clickcounts, clicktimes, choice = trial
    @unpack λ,σ2_a,σ2_s,ϕ,τ_ϕ = θ
    @unpack nT, nL, nR = clickcounts
    @unpack L, R = clicktimes

    #adapt magnitude of the click inputs
    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    P = P0(σ2_i, a₀, n, dx, xc, dt)

    #empty transition matrix for time bins with clicks
    F = zeros(TT,n,n)

    abar = Vector{TT}(undef, nprepad_abar+nT)

    @inbounds for t = 1:nT
        P,F = latent_one_step!(P,F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,M,dx,xc,n,dt)
        abar[nprepad_abar+t] = xcᵀ*P
    end
    abar[1:nprepad_abar] = abar[nprepad_abar+1]

    return P, abar
end

"""
    designmatrix

Make a design matrix from the mean trajectory of the latent variable

=INPUT

-abar: ̅a(t), a vector of vectors of floats indicating the mean of the latent variable. Each element of the outer vector corresponds to a trial, and each element of the inner vector corresponds to a time bin
-a_bases: a vector of vector of floats corresponding to the kernel through which the mean latent trajectory is filtered. Each element of the outer vector corresponds to a basis, and each element of the inner array corresponds to a time bin

=OUTPUT

- a matrix of floats with a number of rows equal to sum of the number of time bins in each trial and a number of columns equal to the number of regressors. The trials are concatenated along the first dimension.

"""
function designmatrix(abar::Vector{Vector{T}}, a_bases::Vector{Vector{T}}) where {T <: AbstractFloat}
    npad = size(a_bases)[1]-1
    vcat(pmap(a->hcat(map(basis->filter(basis, a)[npad+1:end], a_bases)...), abar)...)
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
function mean_square_error(Xtiming::Matrix{T}, Xautoreg::Matrix{T}, Xa:: Matrix{T}, y::Vector{T}, L2regularizer::Matrix{T}) where {T<:AbstractFloat}
    X = hcat(Xtiming, Xautoreg, Xa)
    mean((X*inv(tranpose(X)*X+L2regularizer)*transpose(X)*y-y).^2)
end

"""
    θ2(θ)

Square the values of a subset of parameters (σ2_i,σ2_a, σ2_s)
"""
θ2(θ::θDDLM) = θDDLM(θz=θz2(θ.θz), θh = θ.θh, bias=θ.bias, lapse=θ.lapse)

"""
    invθ2(θ)

Returns the positive square root of a subset of parameters (σ2_i,σ2_a, σ2_s)
"""
invθ2(θ::θDDLM) = θDDLM(θz=invθz2(θ.θz), θh = θ.θh, bias=θ.bias, lapse=θ.lapse)