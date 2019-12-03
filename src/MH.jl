#MH

using AdvancedMH
using Distributions, Random, Parameters
using pulse_input_DDM

pz = [0.5, 10., -0.5, 20., 1.0, 0.6, 0.02]
pd = [1.,0.05]

#data = pulse_input_DDM.sample_clicks_and_choices(pz, pd, 1000)
#data = pulse_input_DDM.bin_clicks!(data)
data = pulse_input_DDM.sample_clicks(20000)

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
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
#density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf
density(choices,dist) = logpdf(dist,choices)

# Generate a set of data from the posterior we want to estimate.
#data = rand(Normal(0, 1), 30)

# Construct a DensityModel.
#model = DensityModel(density, data)

model = DensityModel(logpdf,choices)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000)
