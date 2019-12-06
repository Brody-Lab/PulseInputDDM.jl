using AdvancedMH, pulse_input_DDM, Distributions, Random
using LinearAlgebra

pz, pd, data = default_parameters_and_data(ntrials=100);
model = DensityModel(x-> compute_LL(x,data))

#h = compute_Hessian(pz, pd, data; state="generative");
#h⁻¹ = inv(-1*h)
#F = Matrix(cholesky(Positive, h⁻¹, Val{false}))

#h = 0.5 * (h + h')
#V = eigvecs(h⁻¹)
#d = collect(Diagonal(max.(0, eigvals(h⁻¹))))
#h⁻¹PSD = V * d * V'
#h⁻¹PSD = 0.5 * (h⁻¹PSD + h⁻¹PSD')
#hPSD⁻¹ = inv(hPSD)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

#do hessian for test MH
#do HMC?
#increase support?

#0.1 * [0.1, 10., 1., 10., 0.1, 0.01, 0.01, 0.1, 0.1]
#spl = MALA(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01]));

spl = MALA(vcat(pz["generative"],pd["generative"]), x-> MvNormal(1e-4 * 0.5 * x, [0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01]));

#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.01, 0.001, 1., 0.01]));
#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), [0.1, 1., 0.1, 1., 0.1, 0.1, 0.01, 1., 0.01]));
#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), 0.01 * ones(9)));
#spl = MetropolisHastings(vcat(pz["generative"],pd["generative"]), MvNormal(zeros(9), 0.001 * F));
chain = sample(Random.GLOBAL_RNG, model, spl, 1000; param_names=vcat(pz["name"], pd["name"]))
#chain = sample(model, spl, 200; param_names=["σ_a", "σ_s"])


function ∂ℓπ∂θ(x)
    res = GradientResult(x)
    gradient!(res, x-> compute_LL(x,data), x)
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
θ_init = vcat(pz["generative"],pd["generative"])
#θ_init = zeros(9)

ϵ = 0.05
n_steps = 1

metric = DiagEuclideanMetric(length(θ_init))
h = Hamiltonian(metric, x-> compute_LL(x,data), ∂ℓπ∂θ)
lf = Leapfrog(ϵ)
#lf = Leapfrog(find_good_eps(h, θ_init))
prop = StaticTrajectory(lf, n_steps)
samples, stats = sample(h, prop, θ_init, n_samples; progress=true)
