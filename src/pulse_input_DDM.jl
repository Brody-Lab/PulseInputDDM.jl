module pulse_input_DDM

using StatsBase, Distributions, DSP, Optim, LineSearches, JLD
using ForwardDiff, Distributed, LinearAlgebra
using Pandas
using SpecialFunctions
using MAT, ROCAnalysis

include("latent_DDM_common_functions.jl")
include("helpers.jl")
include("initialize_spike_obs_model.jl")
include("initialize_latent_model.jl")
include("choice_observation.jl")
include("poisson_neural_observation.jl")
include("analysis_functions.jl")
include("choice_and_poisson_neural_observation.jl")
include("run_funcs.jl")

export do_LL, poiss_LL, make_data
export FilterSpikes, nanmean, nanstderr, rate_mat_func_filt
export diffLR, group_by_neuron, opt_ll, do_optim_ΔLR

end # module
