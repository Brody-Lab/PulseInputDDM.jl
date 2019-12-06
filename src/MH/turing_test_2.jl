using LogDensityProblems, Random, DynamicHMC, DynamicHMC.Diagnostics, pulse_input_DDM            # use for AD
import ForwardDiff
using MCMCDiagnostics
using Statistics

f_str, num_sessions, num_trials_per_session, cells_per_session = "softplus", 1, [50], [2]
pz,pd,data = default_parameters_and_data(ntrials=1000)
pz, py, data = default_parameters_and_data(f_str, num_sessions, num_trials_per_session, cells_per_session)
p = DDMProblem(data)
#p = DDMProblem(data)((pz2["generative"], pd2["generative"]))
#θ = (pz2["generative"], pd2["generative"])
#p((pz2["generative"], pd2["generative"]))

#t = problem_transformation(p)
#t = as((B = asℝ₊,))
#P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, TransformedLogDensity(pulse_input_DDM.problem_transformation(p), p));

@time results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000)

# To get the posterior for ``α``, we need to use `get_position` and
# then transform

posterior = transform.(pulse_input_DDM.problem_transformation(p), results.chain);

# Extract the parameter.

posterior_α = first.(posterior);


posterior_bias = mean(last, posterior)

mean(posterior_α)

# check the effective sample size

ess_α = effective_sample_size(posterior_α)

# NUTS-specific statistics

summarize_tree_statistics(results.tree_statistics)
