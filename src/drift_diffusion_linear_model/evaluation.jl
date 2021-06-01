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
    x = pulse_input_DDM.flattent(θ)
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
-choicelikelihood: the predicted likelihood of a right choice in each trial, organized as a vector of vectors. The outer vector corresponds to trial-sets, and the inner array corresponds to trials
-Xa: The columns of the design matrix that depends on the latent variable, organized as a vector of matrics. Each element of the outer array corresponds to a trial-set.
"""

function predict_in_sample(model::DDLM)
    @unpack θ, data, options = model
    @unpack a_bases = options

    options.remap && (θ = θ2(θ))
    output = map(trialset->mean_latent_choice_likelihood(θ, trialset, options), data)
    abar = map(x->x[1], output)
    choicelikelihood = map(x->x[2], output)

    nprepad_abar = size(a_bases[1])[1]-1
    Xa = map(abar->vcat(pmap(a->hcat(map(basis->DSP.filt(basis, a)[nprepad_abar+1:end], a_bases)...), abar)...), abar)

    return abar, choicelikelihood, Xa
end
