using AdvancedMH, Distributions, Random

θtrue = [0.,1.,1.]
#G(θ) = Normal(θ[1], θ[2])
#P_X_Y(θ) = [Truncated(G(θ), 0, Inf), G(θ)]
#P_X_Y(θ) = [G(θ), G(θ)]
#P_X_Y(θ) = [G(θ), Normal(θ[1], θ[3])]
#dist(θ) = Product(P_X_Y(θ))
dist(θ) = MvNormal(θ[1]*ones(2), collect(θ[2:3]))
Random.seed!(1)
data = rand(dist(θtrue), 30)

insupport(θ) = (θ[2] >= 0) && (θ[3] >= 0)
density(θ) = insupport(θ) ? sum(map(i-> logpdf(dist(θ), data[:,i]), 1:size(data,2))) : -Inf

using TransformVariables

t = as((asℝ, asℝ₊, asℝ₊))

density2(θ) = transform_logdensity(t, density, collect(θ))

#using ForwardDiff: hessian
#negdensity(θ) = insupport(θ) ? -1*sum(map(i-> logpdf(dist(θ), data[:,i]), 1:size(data,2))) : Inf
#h = hessian(negdensity, θtrue)
#h = -1 * hessian(density2, zeros(3))
#h⁻¹ = inv(h)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

#using LinearAlgebra

#V = eigvecs(h⁻¹)
#D = eigvals(h⁻¹)

#model = DensityModel(density2)
model = DensityModel(density)

#spl = MetropolisHastings([0.0, 1.0, 100.0], MvNormal([1., 1., 100.]))
#spl = MetropolisHastings([0.0, 1.0, 100.0], MvNormal([1., 1., 1.]))
#spl = MetropolisHastings(zeros(3), MvNormal(h⁻¹))
#spl = MetropolisHastings(zeros(3), MvNormal(h))
#spl = MetropolisHastings(zeros(3), MvNormal(3,0.1))
spl = MetropolisHastings(ones(3), MvNormal(3,0.01));
chain = sample(Random.GLOBAL_RNG, model, spl, 10000; param_names=["μ", "σ1", "σ2"])

#posterior = transform.(t, results.chain);
