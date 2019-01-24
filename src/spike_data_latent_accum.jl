module spike_data_latent_accum

using StatsBase, Distributions, DSP, Optim, LineSearches, JLD
using ForwardDiff, Distributed, LinearAlgebra
using Pandas, SpecialFunctions
#using MAT

include("latent_DDM_common_functions.jl")
include("helpers.jl")
include("initialize_spike_obs_model.jl")
include("initialize_latent_model.jl")
include("choice_observation.jl")
include("poisson_neural_observation.jl")
include("analysis_functions.jl")

export do_LL, poiss_LL
export FilterSpikes, nanmean, nanstderr, rate_mat_func_filt

end # module
