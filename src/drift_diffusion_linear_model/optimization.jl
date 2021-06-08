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

    x0 = pulse_input_DDM.flatten(θ) # x = pulse_input_DDM.flatten(θ)
    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, options, abar, F, P, X) #pulse_input_DDM.loglikelihood(x, data, options, abar, F, P, X)

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
    preallocate(model)

Preallocate memory for variables that will be modified during model fitting

ARGUMENT

model: An instance of the drift-diffusion linear model

RETURN

-abar: mean of the latent variable at each time step in each trial. A vector of vectors of vectors of zero's. Each element of the outermost array corresponds to a trialset, each element of the inner array corresponds to a trial, and each element of the innermost corresponds to a time bin
-F: P(aₜ|aₜ₋₁) A vector of vectors of square matrices of zero's. Each element of the outermost array corresponds to a trialset, and that of the inner array corresponds to a trial. The innermost array is a square matrix of length `n`, where `n` is the number of bins into which latent space is discretized. Each element of the innermost corresponds to the transition for a pair of values (aₜ, aₜ₋₁)
-P: P(aₜ) A vector of vectors of vectors of zero's. Each element of the outermost array corresponds to a trialset, and that of the inner array corresponds to a trial. The innermost array is a square matrix of length `n`, where `n` is the number of bins into which latent space is discretized. Each element of the innermost corresponds to the transition for a a value of aₜ.
-X: The design matrix for least square regression. A vector of vectors of matrices. Each element of the outermost array corresponds to a trialset, each element of the inner array corresponds to a neural unit. Each design matrix has a number of rows equal to the sum of time bins across trials and a number of columns equal to the total number of regressors. The columns corresponding to trial timing and spike train history are pre-established, and the columns corresponding to the latent variable are preallocated to be 0.
"""

function preallocate(model::DDLM)
    @unpack data, options = model
    @unpack a_bases, n, npostpad_abar = options
    nprepad_abar = size(a_bases[1])[1]
    abar = map(trialset->map(trial->fill(0., nprepad_abar+trial.clickindices.nT+npostpad_abar), trialset.trials), data)
    F = map(trialset->fill(fill(0., n, n), length(trialset.trials)), data)
    P = map(trialset->fill(fill(0., n), length(trialset.trials)), data)
    X = map(trialset->map(unit->unit.X, trialset.units), data)
    return abar, F, P, X
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
function loglikelihood(x::Vector{T1}, data::Vector{T2}, options::DDLMoptions, abar, F, P, X) where {T1<:Real, T2<:trialsetdata}
    θ = θDDLM(x)
    options.remap && (θ = θ2(θ))
    latentspec = latentspecification(options, θ)
    sum(map((trialset, abar, F, P, X)->loglikelihood(θ, trialset, latentspec, options, abar, F, P, X), data, abar, F, P, X))
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
    latentspecification(cross=cross, dt=dt, dx=dx, M=M, n=n, nprepad_abar=size(a_bases[1])[1], npostpad_abar=npostpad_abar, type=T1, typedone = one(T1), typedzero=typedzero, xc=xc)
end
"""
    loglikelihood(θ, trialset, latentspec)

Sum the loglikelihood of all trials in a single trial-set

ARGUMENT

-θ: an instance of `θDDLM`
-trialset: an instance of `trialsetdata`
-latentspec: a container of variables specifying the latent space
-`options`: specifications of the drift-diffusion linear model

RETURN

-The summed log-likelihood of the choice and spike trains given the model parameters. Note that the likelihood of the spike trains of all units in each trial is normalized to be of the same magnitude as that of the choice, between 0 and 1.

"""
function loglikelihood(θ::θDDLM, trialset::trialsetdata, latentspec::latentspecification, options::DDLMoptions, abar, F, P, X)

    @unpack a_bases, L2regularizer = options
    @unpack bias, lapse = θ
    @unpack trials, units = trialset
    #latentspec = pulse_input_DDM.latentspecification(options, θ)
    @unpack dx, n, nprepad_abar, xc = latentspec

    P = forwardpass!(abar, F, latentspec, P, θ, trialset) # P[1] = pulse_input_DDM.forwardpass!(abar[1], F[1], latentspec, P[1], θ, data[1])
    ℓℓ_choice = sum(log.(map((P, trial)->sum(choice_likelihood!(bias,xc,P,trial.choice,n,dx)), P, trials))) #ℓℓ_choice = sum(log.(map((P, trial)->sum(pulse_input_DDM.choice_likelihood!(θ.bias,xc,P,trial.choice,n,dx)), P[1], data[1].trials)))
    #Xa = hcat(map(basis->vcat(pmap(abar->DSP.filt(basis, abar)[nprepad_abar+1:end], abar)...), a_bases)...) # Xa = hcat(map(basis->vcat(map(abar->DSP.filt(basis, abar)[nprepad_abar+1:end], abar[1])...), a_bases)...)
    #ℓℓ_spike_train = mean(pmap((unit, X)->mean(loglikelihood(L2regularizer, unit.ℓ₀y, lapse, X, Xa, unit.y)), units, X))*size(trials)[1]
    #ℓℓ_spike_train = mean(map((unit, X)->mean(loglikelihood(L2regularizer, unit.ℓ₀y, θ.lapse, X, Xa, unit.y)), data[1].units, X[1]))*size(data[1].trials)[1]

    #map->pmap
    Xa = hcat(map(basis->vcat(map(abar->DSP.filt(basis, abar)[nprepad_abar+1:end], abar)...), a_bases)...)
    ℓℓ_spike_train = mean(map((unit, X)->mean(loglikelihood(L2regularizer, unit.ℓ₀y, lapse, X, Xa, unit.y)), units, X))*size(trials)[1]

     #ℓℓ_spike_train = mean(map((unit, X)->mean(loglikelihood(L2regularizer, unit.ℓ₀y, θ.lapse, X, Xa, unit.y)), data[1].units, X[1]))*size(data[1].trials)[1]
    ℓℓ_choice + ℓℓ_spike_train
end

"""
    loglikelihood(L2regularizer, ℓ₀y, lapse, X, Xa, y)

Loglikelihood of the spike trains of one neural unit across all trials of a trialset

INPUT

-L2regularizer: penalty matrix to implement L2 regularization. Square matrix of length `p`, where `p` is the total number of regressors. Only the elements of the diagonal are nonzero, and they may have different value
-ℓ₀y: likelihood of the spike train of the specified units given that the latent variable equals 0 for all time bins and all trials. A vector of length `∑T(i)`, where `T(i)` is number of time bins in the i-th trial.
-lapse: The fraction of trials in which the choice is formed independently of the latent variable
-X: the design matrix for least-square regression. A N-by-p matrix, where N is the sum of time bins across trials, and p is the number of regressors
-Xa: the columns of the design matrix corresponding to the filtered output of the mean of the latent variable. It is a N-by-pₐ matrix, where pₐ is the number of filters.
-y: spike train and the response variable of the least-square regression. A vector of length `∑T(i)`, where `T(i)` is number of time bins in the i-th trial. Each element indicates the spike count in a particular time bin of a particular trial.

OUTPUT

-The loglikelihood of the spike trains. A vector of length `∑T(i)`, where `T(i)` is number of time bins in the i-th trial.
"""
function loglikelihood(L2regularizer::Matrix{Float64}, ℓ₀y::Vector{T1}, lapse::T1, X::Matrix{T1}, Xa::Matrix{T1}, y::Vector{Float64}) where {T1<:Real}
    nbases = size(Xa)[2]
    X[:, end-nbases+1:end] = Xa # X[1][1][:, end-nbases+1:end] = Xa
    β = inv(transpose(X)*X+L2regularizer)*transpose(X)*y # β = inv(transpose(X[1][1])*X[1][1]+L2regularizer)*transpose(X[1][1])*data[1].units[1].y
    ŷ = X*β # ŷ = X[1][1]*β
    e = y-ŷ # e = data[1].units[1].y-ŷ
    # using Statistics
    σ² = var(e)
    log.((((1-lapse)/sqrt(2π*σ²)).*exp.(-(e.^2)./2σ²) + lapse.*ℓ₀y)) #log.((((1-θ.lapse)/sqrt(2π*σ²)).*exp.(-(e.^2)./2σ²) + θ.lapse.*data[1].units[1].ℓ₀y))
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
function forwardpass!(abar::Any, F::Any, latentspec::latentspecification, P, θ::θDDLM, trialset::trialsetdata)
    @unpack α, B, k = θ
    @unpack lagged, trials = trialset

    a₀ = pulse_input_DDM.history_influence_on_initial_point(α, B, k, lagged)
    #pmap((abar, F, a₀, P, trial)->forwardpass!(abar, F, a₀, latentspec, P, θ, trial), abar, F, a₀, P, trials)

    #map -> pmap
    map((abar, F, a₀, P, trial)->forwardpass!(abar, F, a₀, latentspec, P, θ, trial), abar, F, a₀, P, trials)
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
function forwardpass!(abar::Vector{T1}, F::Matrix{T1}, a₀::Any, latentspec::latentspecification, P::Vector{T1}, θ::θDDLM, trial::trialdata) where {T1<:Real}
    @unpack clickindices, clicktimes, choice = trial
    @unpack σ2_i, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θ
    @unpack nT, nL, nR = clickindices
    @unpack L, R = clicktimes
    @unpack cross, dt, dx, n, nprepad_abar, type, typedzero, typedone, xc = latentspec

    if typeof(F[1])==type
        abar.= typedzero
        F .= typedzero
        P .= typedzero
    else
        abar = abar.*typedzero
        F = F.*typedzero #F[1][1] = F[1][1].*typedzero
        P = P.*typedzero # P[1][1] = P[1][1].*typedzero
        println("=====constructed====")
    end
    P[ceil(Int,n/2)] = typedone #P[1][1][ceil(Int,n/2)] = typedone
    transition_M!(F, σ2_i, typedzero, a₀, dx, xc, n, dt) #pulse_input_DDM.transition_M!(F[1][1], σ2_i, typedzero, a₀[1], dx, xc, n, dt)
    P = F * P #P[1][1] = F[1][1] * P[1][1]

    La, Ra = adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross) #La, Ra = pulse_input_DDM.adapt_clicks(ϕ,τ_ϕ,L,R; cross=cross)

    @inbounds for t = 1:nT
        P = forward_one_step!(F,λ,σ2_a,σ2_s,t,nL,nR,La,Ra,latentspec,P)
        abar[nprepad_abar+t] = sum(xc.*P)
    end
    # @inbounds for t = 1:nT
    #     P[1][1] = pulse_input_DDM.forward_one_step!(F[1][1],λ,σ2_a,σ2_s,t,nL,nR,La,Ra,latentspec,P[1][1])
    #     abar[1][1][nprepad_abar+t] = sum(xc.*P[1][1])
    # end

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

    @unpack dt, dx, M, n, typedzero, xc = latentspec

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
    # any(t .== nL) ? sL = sum(La[t .== nL]) : sL = typedzero
    # any(t .== nR) ? sR = sum(Ra[t .== nR]) : sR = typedzero
    # sLR = sL + sR
    # if sLR > typedzero
    #     σ2 = σ2_s*sLR + σ2_a*dt
    #     pulse_input_DDM.transition_M!(F[1][1], σ2, λ, sR-sL, dx, xc, n, dt)
    #     P[1][1]=F[1][1] * P[1][1]
    # else
    #     P[1][1]=P[1][1] = M*P[1][1]
    # end
    # sum(P[1][1])
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
"""
function θ2(θ::θDDLM)
    @unpack θz, θh, bias, lapse = θ
    x = pulse_input_DDM.flatten(θz)
    index = convert(BitArray, map(x->occursin("σ2", string(x)), collect(fieldnames(θz))))
    x[index]=x[index].^2
    θDDLM(θz=θz(x), θh=θh, bias=bias, lapse=lapse)
end
