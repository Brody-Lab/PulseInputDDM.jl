using LogDensityProblems, DynamicHMC, DynamicHMC.Diagnostics
using MCMCDiagnostics
@everywhere using Statistics, Random
import ForwardDiff              # use for AD
@everywhere using pulse_input_DDM

pz2,pd2,data = default_parameters_and_data(ntrials=1000)

p = DDMProblem(data)((pz2["generative"], pd2["generative"]))
#θ = (pz2["generative"], pd2["generative"])
#p((pz2["generative"], pd2["generative"]))

#t = problem_transformation(p)
#t = as((B = asℝ₊,))
#P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, TransformedLogDensity(problem_transformation(p), p));

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
