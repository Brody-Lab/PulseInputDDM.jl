#using Turing
#using DynamicHMC
#using Distributions: logpdf, MvNormal

using Turing, StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  #s ~ InverseGamma(2, 3)
  s ~ Product(Uniform.(zeros(length(x)), Inf))
  for i in eachindex(x)
    m ~ Normal(0, sqrt(s[i]))
    x[i] ~ Normal(m, sqrt(s[i]))
  end
end

#  Run sampler, collect results
chn = sample(gdemo(randn(100)), HMC(0.1, 5), 1000)

#=
@model mymodel() = begin
  d = 10 # the actual dimension of your problem
  p ~ DummayPrior(d)
  1.0 ~ Target(p)
end

struct DummayPrior <: ContinuousMultivariateDistribution
  d # the dimension of your problem
end

Base.length(dp::DummayPrior) = dp.d
Distributions.logpdf(::DummayPrior, _::Vector{T} where {T}) = 0.0
Turing.init(dp::DummayPrior) = rand(dp.d)
Distributions.rand(dp::DummayPrior) = randn(dp.d)

struct Target <: ContinuousMultivariateDistribution
  p
end

target = MvNormal(zeros(dp.d), ones(dp.d))
ℓπ(θ) = logpdf(target, θ)

Distributions.logpdf(t::Target, _) = ℓπ(t.p)

mf = mymodel()
chn = sample(mf, DynamicNUTS(), 2000)
# If our_log_target is differentiable, you can also use HMC or NUTS, e.g.
# alg = HMC(1_000, 0.2, 3) # HMC for 1_000 samples with 0.2 as step size an

=#

#chn = sample(mf, DynamicNUTS(), 2000)


@model model(x) = begin
  #s ~ InverseGamma(2,3)
  s ~ Product(Uniform.(zeros(size(x,2)), ones(size(x,2))))
  m ~ Normal.(0,sqrt(s))
  [x[i] ~ Normal(m, sqrt(s)) for i in eachindex(x)]
  return s, m
end
#chn = sample(gdemo([1.5, 2]), MH(100, (:m, (x) -> Normal(x, 0.1))))
chn = sample(model(randn(10,2)), HMC(0.1, 5), 10000)
