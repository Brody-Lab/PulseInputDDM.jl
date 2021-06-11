"""
    Hessian(model; chunck_size, remap)

Compute the hessian of the negative log-likelihood at the current value of the parameters of a `DDLM`.

Arguments:

- `model`: instance of [`jointDDM`](@ref)

Optional arguments:

- `chunk_size`: parameter to manange how many passes over the LL are required to compute the Hessian. Can be larger if you have access to more memory.
- `remap`: For considering parameters in variance of std space.

"""
function Hessian(model::DDLM; chunk_size::Int=4)

    @unpack θ, data, options = model
    @unpack fit = options
    x = pulse_input_DDM.flatten(θ)
    x,c = unstack(x, fit)
    ℓℓ(x) = -loglikelihood(stack(x,c,fit), data, options)
    cfg = ForwardDiff.HessianConfig(ℓℓ, x, ForwardDiff.Chunk{chunk_size}())
    ForwardDiff.hessian(ℓℓ, x, cfg)
end

"""
    predict_in_sample

Compute the model-predicted choice probability and mean of the latent variable in each trial in all trial-sets

ARGUMENT
-model: an instance of `DDLM,` a drift-diffusion linear model

RETURN
-abar: the predicted mean of the latent variable, organized as a vector of vectors of vectors. The outermost array corresponds to trial-sets, the second inner array coresponds to trials, and the innermost aray coresponds to time bins
-β: regressor coefficients, organized as a vector of vectors of vectors. The outermost array corresponds to trial-sets, the second inner array coresponds to units, and the innermost aray corresponds to regressors
-choicelikelihood: the predicted likelihood of a right choice in each trial, organized as a vector of vectors. The outer vector corresponds to trial-sets, and the inner array corresponds to trials
-ŷ: predicted spike train, concatenated across all trials of a trialset. Organized as a vector of vectors of vectors. The outermost array corresponds to trial-sets, the second inner array coresponds to units, and the innermost aray corresponds to a time bin in a trial
"""

function predict_in_sample(model::DDLM)
    @unpack θ, data, options = model
    @unpack bias, σ2_a = θ
    @unpack a_bases, autoreg_bases = options

    options.remap && (θ = θ2(θ))
    latentspec = pulse_input_DDM.latentspecification(options, θ)
    @unpack dx, n, nprepad_abar, npostpad_abar, xc = latentspec

    abar = map(trialset->map(trial->fill(zero(σ2_a), nprepad_abar+trial.clickindices.nT+npostpad_abar), trialset.trials), data)
    P = map((abar, trialset)->forwardpass!(abar, latentspec, θ, trialset), abar, data)
    choicelikelihood = map((P,trialset)->pmap((P, trial)->sum(choice_likelihood!(bias,xc,P,trial.choice,n,dx)), P, trialset.trials), P, data)

    nprepad_abar = size(a_bases[1])[1]
    Xa = map(abar->hcat(map(basis->vcat(pmap(abar->DSP.filt(basis, abar)[nprepad_abar+1:end], abar)...), a_bases)...), abar)
    βcoupled = map((trialset, Xa)->map(unit->leastsquares(unit, Xa, trialset.Xtiming), trialset.units), data, Xa)
    β = map((βcoupled, coupling, trialset)->map((βcoupled, coupling, unit)->βcoupled.*coupling + unit.βuncoupled.*(1-coupling), βcoupled, coupling, unit), βcoupled, coupling, trialset.units)
    ŷ = map((β, trialset, Xa)->map(β->predict_spike_train(autoreg_bases, β, trialset.nbins_each_trial, Xa, trialset.Xtiming), β), β, data, Xa)

    return abar, β, choicelikelihood, Xa, ŷ
end
