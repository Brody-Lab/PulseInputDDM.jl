module pulse_input_DDM

using StatsBase, Distributions, Optim, LineSearches, JLD2
using ForwardDiff, Distributed, LinearAlgebra
using SpecialFunctions, MAT, Random 
using DSP
import Pandas: qcut

#using ROCAnalysis
#using ImageFiltering
#using DataFrames
#using BasisFunctionExpansions

#include("initialize_spike_obs_model.jl")
include("latent_variable_model_functions.jl")
include("helper_functions.jl")
include("analysis_functions.jl")
include("choice_and_poisson_neural_observation.jl")
include("data_sessions.jl")

include("wrapper_functions.jl")
include("mapping_functions.jl")
include("sample_model_functions.jl")

include("choice_model/choice_observation_model.jl")
include("choice_model/wrapper_functions.jl")
include("choice_model/mapping_functions.jl")
include("choice_model/sample_model_functions.jl")
include("choice_model/manipulate_data_functions.jl")

export aggregate_choice_data, bin_clicks!
export sample_choices!, sample_inputs_and_choices

include("neural_model/poisson_neural_observation.jl")
include("neural_model/wrapper_functions.jl")
include("neural_model/mapping_functions.jl")
include("neural_model/sample_model_functions.jl")
include("neural_model/manipulate_data_functions.jl")

export compute_H_CI!, optimize_model, compute_LL, load_and_optimize, compute_Hessian

export poiss_LL, aggregate_spiking_data
export nanmean, nanstderr
export diffLR, dimz
export fy, bins, sigmoid_4param, softplus_3param
export filter_data_by_cell!, sessids_from_region, group_by_neuron!, aggregate_and_append_extended_spiking_data!
export train_test_divide

end
