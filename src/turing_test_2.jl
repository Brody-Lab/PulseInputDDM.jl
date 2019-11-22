using TransformVariables, LogDensityProblems, DynamicHMC, DynamicHMC.Diagnostics
using MCMCDiagnostics
using Parameters, Statistics, Random
import ForwardDiff              # use for AD
using pulse_input_DDM

pz2,pd2,data = default_parameters_and_data(ntrials=10)

struct DDMProblem
    data::Dict{Any,Any}
end

#=
mutable struct latent_params
    σ2_i::Float64
    B::Float64
    λ::Float64
    σ2_a::Float64
    σ2_s::Float64
    ϕ::Float64
    τ_ϕ::Float64
end

mutable struct observation_params
    bias::Float64
    lapse::Float64
end
=#

# Then make the type callable with the parameters *as a single argument*.  We
# use decomposition in the arguments, but it could be done inside the function,
# too.

function (problem::DDMProblem)(θ)
    pz, pd = θ
    @unpack data = problem      # extract the data
    ## log likelihood: the constant log(combinations(n, s)) term
    ## has been dropped since it is irrelevant to sampling.
    #s * log(α) + (n-s) * log(1-α)
    compute_LL(collect(pz), collect(pd), data)
end

# We should test this, also, this would be a good place to benchmark and
# optimize more complicated problems.

#pz = latent_params(pz2["generative"]...)
#pd = observation_params(pd2["generative"]...)
p = DDMProblem(data)
θ = (pz2["generative"], pd2["generative"])
p(θ)

function problem_transformation(problem::DDMProblem)
    #pz, pd = θ
    #σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
    #bias, lapse = pd
    #tz = as((as(Real, 0., 2.), as(Real, 8., 30.),
#            as(Real, -5, 5), as(Real, 0., 100.), as(Real, 0., 2.5),
    #        as(Real, 0.01, 1.2), as(Real, 0.005, 1.)))
    #td = as((as(Real, -30., 30.), as(Real, 0., 1.)))

    tz = as((as(Real, 0., 2.), as(Real, 8., 30.),
            as(Real, -5, 5), as(Real, 0., 100.), as(Real, 0., 2.5),
            as(Real, 0.01, 1.2), as(Real, 0.005, 1.)))

    td = as((as(Real, -30., 30.), as(Real, 0., 1.)))

    as((tz,td))

end

t = problem_transformation(p)
#t = as((B = asℝ₊,))
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P);

# Finally, we sample from the posterior. `chain` holds the chain (positions and
# diagnostic information), while the second returned value is the tuned sampler
# which would allow continuation of sampling.

results2 = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000)

# To get the posterior for ``α``, we need to use `get_position` and
# then transform

posterior = transform.(t, results.chain);

# Extract the parameter.

posterior_α = first.(posterior);


posterior_bias = mean(last, posterior)

mean(posterior_α)

# check the effective sample size

ess_α = effective_sample_size(posterior_α)

# NUTS-specific statistics

summarize_tree_statistics(results.tree_statistics)
