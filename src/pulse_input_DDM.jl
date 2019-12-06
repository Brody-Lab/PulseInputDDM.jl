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
using Roots
using TransformVariables, Parameters
import Base.rand
import Distributions: logpdf
using Base.Threads

include("MH/MCMC.jl")

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

export dimz
export compute_CIs!, optimize_model, compute_LL, compute_Hessian, compute_gradient
export default_parameters, LL_all_trials
export bin_clicks!, load_choice_data, bounded_mass_all_trials
export reload_optimization_parameters, save_optimization_parameters
export default_parameters_and_data
export LL_across_range

export DDMProblem, choiceDDM, latent, choice, inputs
export logpdf

export mean_exp_rate_per_trial, mean_exp_rate_per_cond

#=

export neural_null
export compute_ΔLL

export choice_null
export sample_input_and_spikes_multiple_sessions, sample_inputs_and_spikes_single_session
export sample_spikes_single_session, sample_spikes_single_trial, sample_expected_rates_single_session

export sample_choices_all_trials
export aggregate_spiking_data, bin_clicks_spikes_and_λ0!

export diffLR

export filter_data_by_cell!

=#

end
