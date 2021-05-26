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
    x = vec(θ)
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
-choiceprobability: the predicted probability of a right choice in each trial, organized as a vector of vectors. The outer vector corresponds to trial-sets, and the inner array corresponds to trials
"""

function predict_in_sample(model::DDLM)
    @unpack θ, data, options = model
    options.remap && (θ = θ2(θ))
    map(trialset->predict_in_sample(trialset, θ, options), data)
end

"""
    predict_in_sample

Compute the model-predicted choice probability and mean of the latent variable in each trial in a single trial-set

ARGUMENT
-trialset: an instance of `trialsetdata`
-θ: an instance of `θDDLM,` parameters of the drift-diffusion linear model
-options: an instance of `DDLMoptions,` specification of the of the drift-diffusion linear model

RETURN
-abar: the predicted mean of the latent variable, organized as a vector of vectors. The outer array coresponds to trials, and the inner aray coresponds to time bins
-choiceprobability: the predicted probability of a right choice in each trial, organized as a vector of floats.
"""

function predict_in_sample(trialset::trialsetdata, θ::θDDLM, options::DDLMoptions)
    @unpack θz, θh, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a = θz
    @unpack α, k = θh
    @unpack trials, shifted, units = trialset
    @unpack a_bases, n, cross = options

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt) # P is not used
    a₀ = history_influence_on_initial_point(θh, B, shifted)
    P, abar = pmap((trial,a₀)->latent_one_trial(θz, trial, a₀, M, transpose(xc), dx, n, cross, 0), trials, a₀)
    choiceprobability = pmap((P, trial)->sum(choice_likelihood!(bias,xc,P,true,n,dx)) * (1 - lapse) + lapse/2, P, trials)

    return abar, choiceprobability
end
