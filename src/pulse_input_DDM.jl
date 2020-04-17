"""
    pulse_input_DDM

A julia module for fitting bounded accumlator models using behavioral and/or neural data from pulse-based evidence accumlation tasks.
"""
module pulse_input_DDM

using StatsBase, Distributions, LineSearches, JLD2
using ForwardDiff, Distributed, LinearAlgebra
using Optim, DSP
using SpecialFunctions, MAT, Random
using Discretizers
import StatsFuns: logistic, logit, softplus, xlogy
using ImageFiltering
using ForwardDiff: value

include("base_model.jl")
include("analysis_functions.jl")
include("optim_funcs.jl")
include("sample_model.jl")

include("choice_model/compute_LL.jl")
include("choice_model/wrappers.jl")
include("choice_model/sample_model.jl")
include("choice_model/process_data.jl")

include("neural_model/compute_LL.jl")
include("neural_model/wrappers.jl")
include("neural_model/sample_model.jl")
include("neural_model/process_data.jl")
include("neural_model/deterministic_model.jl")

#include("neural_model/load_and_optimize.jl")
#include("neural_model/sample_model_functions_FP.jl")

export dimz, RTfit
export sample_clicks_and_choices, sample_choices_all_trials
export split_variable_and_const, combine_latent_and_observation, split_latent_and_observation, combine_variable_and_const
export compute_CIs!, optimize_model, compute_LL, compute_Hessian, compute_gradient
export default_parameters, LL_all_trials
export bin_clicks!, load_choice_data
export reload_optimization_parameters, save_optimization_parameters
export default_parameters_and_data
export LL_across_range

export mean_exp_rate_per_trial, mean_exp_rate_per_cond

export diffLR

#=

export neural_null
export compute_ΔLL

export choice_null
export sample_input_and_spikes_multiple_sessions, sample_inputs_and_spikes_single_session
export sample_spikes_single_session, sample_spikes_single_trial, sample_expected_rates_single_session

export sample_choices_all_trials
export aggregate_spiking_data, bin_clicks_spikes_and_λ0!


export filter_data_by_cell!

=#

end
