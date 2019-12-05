using AdvancedMH, pulse_input_DDM, Distributions, Random

pz, pd, data = default_parameters_and_data(ntrials=1000);
model = DensityModel(x-> compute_LL(x,data))

h = compute_Hessian(pz,pd, data; state="generative");
h⁻¹ = inv(h)
#h = 0.5 * (h + h')
V = eigvecs(h⁻¹)
d = collect(Diagonal(max.(0,eigvals(h⁻¹))))
h⁻¹PSD = V * d * V'
#hPSD⁻¹ = inv(hPSD)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), 0.1 * [0.1, 1., 1., 10., 0.1, 0.01, 0.01, 0.1, 0.1]));
chain = sample(Random.GLOBAL_RNG, model, spl, 1000; param_names=vcat(pz["names"],pd["names"]))
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
