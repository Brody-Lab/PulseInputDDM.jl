#MH

using Distributions, Random, Parameters
using pulse_input_DDM
using Distributions: logpdf, MvNormal
using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

pz = [0.5, 10., -0.5, 20., 1.0, 0.6, 0.02]
pd = [1.,0.05]
npts = 200

#data = pulse_input_DDM.sample_clicks_and_choices(pz, pd, 1000)
#data = pulse_input_DDM.bin_clicks!(data)
data = pulse_input_DDM.sample_clicks(npts)

#pz, pd, data = default_parameters_and_data(ntrials=1000);
pz2 = latent(pz...)
pd2 = choice(pd...)

dt, n = 1e-2, 53

T, L, R = data["T"], data["leftbups"], data["rightbups"]
binned = map((T,L,R)-> pulse_input_DDM.bin_clicks(T,L,R; dt=dt), data["T"], data["leftbups"], data["rightbups"])
nT, nL, nR = map(x->getindex.(binned, x), 1:3)

#I = inputs.(L, R, T, nT, nL, nR, dt)
I = inputs(L, R, T, nT, nL, nR, dt)

#dist = map(i-> choiceDDM(pz2, pd2, i), I);

#end
dist = choiceDDM(pz2, pd2, I);

#@time LL_all_trials(pz["generative"], pd["generative"], data)
#@time pulse_input_DDM.LL_all_trials_thread(pz["generative"], pd["generative"], data)

#data = rand(dist, Nobs)
#need dtMC, not dt
#@everywhere choices = rand.(dist)
choices = rand(dist)

#logpdf.(dist, choices)

logpdf(dist,choices)

# Define the components of a basic model.
#insupport(θ) = θ[2] >= 0
#dist(θ) = Normal(θ[1], θ[2])
#density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf
function ℓπ(θ)
    pz = latent(θ[1:7]...)
    pd = choice(θ[8:9]...)
    dist = choiceDDM(pz, pd, I);
    logpdf(dist, choices)
end

function ∂ℓπ∂θ(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end

### Build up a HMC sampler to draw samples
using AdvancedHMC

# Sampling parameter settings
n_samples = 200
n_adapts = 2_000

# Draw a random starting points
θ_init = vcat(pz,pd)

ϵ = 0.1
n_steps = 20

# Define metric space, Hamiltonian, sampling method and adaptor
metric = DiagEuclideanMetric(length(θ_init))
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
#int = Leapfrog(find_good_eps(h, θ_init))
lf = Leapfrog(ϵ)
#prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
prop = StaticTrajectory(lf, n_steps)
adaptor = StanHMCAdaptor(
    n_adapts,
    Preconditioner(metric),
    NesterovDualAveraging(0.8, int)
)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, prop, θ_init, n_samples; progress=true)
