#MH

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
#pz2 = latent(pz...)
#pd2 = choice(pd...)

dt, n = 1e-2, 53

#T, L, R = data["T"], data["leftbups"], data["rightbups"]
#binned = map((T,L,R)-> pulse_input_DDM.bin_clicks(T,L,R; dt=dt), data["T"], data["leftbups"], data["rightbups"])
#nT, nL, nR = map(x->getindex.(binned, x), 1:3)

#I = inputs.(L, R, T, nT, nL, nR, dt)
#I = inputs(L, R, T, nT, nL, nR, dt)

#dist = map(i-> choiceDDM(pz2, pd2, i), I);

#end
#dist = choiceDDM(pz2, pd2, I);

#@time LL_all_trials(pz["generative"], pd["generative"], data)
#@time pulse_input_DDM.LL_all_trials_thread(pz["generative"], pd["generative"], data)

#data = rand(dist, Nobs)
#need dtMC, not dt
#@everywhere choices = rand.(dist)
#choices = rand(dist)

#logpdf.(dist, choices)

#logpdf(dist,choices)


# Define the components of a basic model.
#insupport(θ) = θ[2] >= 0
#dist(θ) = Normal(θ[1], θ[2])
#density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf
#function density(x)
#    pz = latent(x[1:7]...)
#    pd = choice(x[8:9]...)
#    dist = choiceDDM(pz, pd, I);
#    logpdf(dist, choices)
#end

#density3(x) = pulse_input_DDM.density2(x, pz, pd, L, R, nT, nL, nR, choices; n=n, dt=dt)
#density3(x) = pulse_input_DDM.density4(x, L, R, nT, nL, nR, choices; n=n, dt=dt)

#pulse_input_DDM.problem_transformation(tuple(pz...,pd...))

#LLs = collect(range(1e-12,stop=100-1e-12,length=20))
#LL = Vector{Float64}(undef,length(LLs))

#for j = 1:length(LLs)
#    LL[j] = density3([LLs[j],pz[5]])
#end

#t = as((as(Real, 0., 2.), as(Real, 2., 30.),
#        as(Real, -5, 5), as(Real, 0., 100.), as(Real, 0., 2.5),
#        as(Real, 0.01, 1.2), as(Real, 0.005, 1.), as(Real, -10., 10.), as(Real, 0., 1.)))

#(0. < σ2_i < 2.) && (2. < B < 30.) && (-5. < λ < 5.) &&
#(0. < σ2_a < 100.) && (0. < σ2_s < 2.5) && (0.01 < ϕ < 1.2) &&
#(0.005 < τ_ϕ < 1.) && (-10. < bias < 10.) && (0. < lapse < 1.)

#x = tuple(pz...,pd...)

#blah(x) = transform_and_logjac(t, collect(x))

#density5(x) = transform_logdensity(t, density3, collect(x))

#x = (1.,2.)
#t = as((as(Real, 0., 2.), as(Real, 8., 30.)))
#f(x) = logpdf(MvNormal(length(x),1.), collect(x))
#transform_logdensity(t, f, collect(x))

# Generate a set of data from the posterior we want to estimate.
#data = rand(Normal(0, 1), 30)

# Construct a DensityModel.
#model = DensityModel(density, data)

#model = DensityModel(density, choices)
#model = DensityModel(density)
model = DensityModel(density3)

using ForwardDiff: hessian
h = -1 * hessian(density3, vcat(pz,pd))
#h⁻¹ = inv(h)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

# Set up our sampler with initial parameters.
#conditionals = [Uniform(0.,2.), Uniform(2.,30.), Uniform(-5.,5.),
#                    Uniform(0.,100.), Uniform(0.,2.5), Uniform(0.01,1.2),
#                    Uniform(0.005,1.), Uniform(-10.,10.), Uniform(0,1)]

#conditionals = [Uniform(0.,100.),Uniform(0.,2.5)]

#spl = MetropolisHastings(vcat(pz,pd), Product(conditionals))
#spl = MetropolisHastings([pz[4]], Product(conditionals))
#spl = MetropolisHastings(vcat(pz,pd))
#spl = MetropolisHastings(pz[4:5])

t = as((as(Real, 0., 2.), as(Real, 2., 30.),
        as(Real, -5, 5), as(Real, 0., 100.), as(Real, 0., 2.5),
        as(Real, 0.01, 1.2), as(Real, 0.005, 1.), as(Real, -10., 10.), as(Real, 0., 1.)))

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
