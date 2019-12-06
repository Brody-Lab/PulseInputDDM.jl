using AdvancedMH, Distributions, Random

θtrue = [0.,1.]
dist(θ) = MvNormal(θ, ones(2))
#dist(θ) = MvNormal(θ[1]*ones(2), θ[2])
data = rand(dist(θtrue), 30)

ℓπ(θ) = logpdf(target, θ)

insupport(θ) = (θ[2] >= 0)
#density(θ) = insupport(θ) ? sum(map(i-> logpdf(dist(θ), data[:,i]), 1:size(data,2))) : -Inf
density(θ) = logpdf(dist(θtrue), θ)

model = DensityModel(density)

τ = 0.8
spl_MALA = MALA(θtrue, x-> MvNormal(τ * 0.5 * x, τ))

τ_MH = 1
spl_MH = MetropolisHastings(θtrue, MvNormal(2, τ_MH));

nsamps = 1000
chain_MALA = sample(Random.GLOBAL_RNG, model, spl_MALA, nsamps; param_names=["μ", "σ"])
chain_MH = sample(Random.GLOBAL_RNG, model, spl_MH, nsamps; param_names=["μ", "σ"])

using PyPlot

plot(vec(chain_MH["σ"].value))
plot(vec(chain_MALA["σ"].value))

1. - sum(isapprox.(sum(diff(chain_MH.value.data[:,2:3,1],dims=1), dims=2), 0.))/ nsamps
1. - sum(isapprox.(sum(diff(chain_MALA.value.data[:,2:3,1],dims=1), dims=2), 0.))/ nsamps
