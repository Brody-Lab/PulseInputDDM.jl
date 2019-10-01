module pulse_input_DDM

using StatsBase, Distributions, Optim, LineSearches, JLD2
using ForwardDiff, Distributed, LinearAlgebra
using SpecialFunctions, MAT, Random
using DSP, Discretizers
import StatsFuns: logistic, logit, softplus
using Base.Threads, ImageFiltering
import Base.convert

#include("initialize_spike_obs_model.jl")
include("latent_variable_model_functions.jl")
include("analysis_functions.jl")
include("choice_and_poisson_neural_observation.jl")
include("data_sessions.jl")

include("optim_funcs.jl")
include("mapping_functions.jl")
include("sample_model_functions.jl")

include("choice_model/choice_observation_model.jl")
include("choice_model/wrapper_functions.jl")
include("choice_model/mapping_functions.jl")
include("choice_model/sample_model_functions.jl")
include("choice_model/manipulate_data_functions.jl")

include("choice_model/dx/wrapper_functions.jl")
include("choice_model/marino/wrapper_functions.jl")
include("choice_model/marino/choice_observation_model.jl")
include("choice_model/marino/mapping_functions.jl")
include("choice_model/marino/sample_model_functions.jl")
include("choice_model/marino/manipulate_data_functions.jl")

export aggregate_choice_data, bin_clicks!
export sample_choices_all_trials!, sample_inputs_and_choices
export dimz

include("neural_model/poisson_neural_observation.jl")
include("neural_model/wrapper_functions.jl")
include("neural_model/mapping_functions.jl")
include("neural_model/sample_model_functions.jl")
include("neural_model/manipulate_data_functions.jl")
include("neural_model/load_and_optimize.jl")
include("neural_model/deterministic_model.jl")
include("neural_model/sample_model_functions_FP.jl")

export compute_H_CI!, optimize_model, compute_LL, load_and_optimize, compute_Hessian
export neural_null
export regress_init, init_pz_py, optimize_and_errorbars, compute_ΔLL
export optimize_model_dx, compute_H_CI_dx!, compute_LL_dx, compute_gradient_dx

export choice_null

export compute_LL_and_prior
export sample_input_and_spikes_multiple_sessions, sample_inputs_and_spikes_single_session
export sample_spikes_single_session, sample_spikes_single_trial, sample_expected_rates_single_session

export sample_choices_all_trials, load_choice_data
export aggregate_spiking_data, bin_clicks_spikes_and_λ0!

export diffLR, rate_mat_func_filt, nanmean, nanstderr

export filter_data_by_cell!, sessids_from_region

end
