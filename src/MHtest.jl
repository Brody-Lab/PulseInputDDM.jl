using AdvancedMH, Distributions

θtrue = [0,1,100]
G(θ) = Normal(θ[1], θ[2])
#P_X_Y(θ) = [Truncated(G(θ), 0, Inf), G(θ)]
#P_X_Y(θ) = [G(θ), G(θ)]
P_X_Y(θ) = [G(θ), Normal(θ[1], θ[3])]
dist(θ) = Product(P_X_Y(θ))
data = rand(dist(θtrue), 30)

insupport(θ) = (θ[2] >= 0) && (θ[3] >= 0)
density(θ) = insupport(θ) ? sum(map(i-> logpdf(dist(θ), data[:,i]), 1:size(data,2))) : -Inf
negdensity(θ) = insupport(θ) ? -1*sum(map(i-> logpdf(dist(θ), data[:,i]), 1:size(data,2))) : Inf

using ForwardDiff: hessian

h = hessian(negdensity, θtrue)
h⁻¹ = inv(h)
#h⁻¹ = 0.5 * (h⁻¹ + h⁻¹')

#using LinearAlgebra

#V = eigvecs(h⁻¹)
#D = eigvals(h⁻¹)

model = DensityModel(density)

#spl = MetropolisHastings([0.0, 1.0, 100.0], MvNormal([1., 1., 100.]))
#spl = MetropolisHastings([0.0, 1.0, 100.0], MvNormal([1., 1., 1.]))
spl = MetropolisHastings([0.0, 1.0, 100.0], MvNormal(h⁻¹))
#spl = MetropolisHastings(θtrue, MvNormal(h))

chain = sample(model, spl, 100000; param_names=["μ", "σ1", "σ2"])
