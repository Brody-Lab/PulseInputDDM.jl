using AdvancedMH
using Distributions, Random, Parameters
using pulse_input_DDM
using TransformVariables

#pz = [0.5, 10., -0.5, 20., 1.0, 0.6, 0.02]
#pd = [1.,0.05]

#npoints = 1000

#data = pulse_input_DDM.sample_clicks_and_choices(pz, pd, 1000)
#data = pulse_input_DDM.bin_clicks!(data)
#data = pulse_input_DDM.sample_clicks(npoints)

pz, pd, data = default_parameters_and_data(ntrials=1000);
model = DensityModel(compute_LL)

h = -1 * compute_Hessian(pz, pd, data);
#h⁻¹ = inv(h)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

spl = MetropolisHastings(vcat(pz,pd), MvNormal(zeros(9), 0.1 * [0.1, 1., 1., 10., 0.1, 0.01, 0.01, 0.1, 0.1]));
chain = sample(model, spl, 1000; param_names=["σ_i", "B", "λ", "σ_a", "σ_s", "ϕ", "τ_ϕ", "bias", "lapse"])
#chain = sample(model, spl, 200; param_names=["σ_a", "σ_s"])


function ∂ℓπ∂θ(x)
    res = GradientResult(x)
    gradient!(res, density5, x)
    return (value(res), gradient(res))
end

### Build up a HMC sampler to draw samples
using AdvancedHMC
using Distributions: logpdf, MvNormal
using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

# Sampling parameter settings
n_samples = 20

# Draw a random starting points
#θ_init = vcat(pz, pd)
θ_init = zeros(9)

ϵ = 0.05
n_steps = 1

metric = DiagEuclideanMetric(length(θ_init))
h = Hamiltonian(metric, density5, ∂ℓπ∂θ)
lf = Leapfrog(ϵ)
#lf = Leapfrog(find_good_eps(h, θ_init))
prop = StaticTrajectory(lf, n_steps)
samples, stats = sample(h, prop, θ_init, n_samples; progress=true)
