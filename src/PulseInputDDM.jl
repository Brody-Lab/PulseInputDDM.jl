"""
    Pulse Input DDM

A julia module for fitting bounded accumlator models using behavioral
and/or neural data from pulse-based evidence accumlation tasks.
"""

module PulseInputDDM

using StatsBase, Distributions, LineSearches
using ForwardDiff, Distributed, LinearAlgebra
using Optim, DSP, SpecialFunctions, MAT, Random
using Discretizers, ImageFiltering
using ForwardDiff: value
using PositiveFactorizations, Parameters, Flatten
using Polynomials, Missings
using HypothesisTests, TaylorSeries
using BasisFunctionExpansions

import StatsFuns: logistic, logit, softplus, xlogy
import Base.rand
import Base.Iterators: partition
import Flatten: flattenable
#import Polynomials: Poly

export choiceDDM, θchoice, θz, choiceoptions
export choiceDDM_dx
export neuralDDM, θneural, θy, neural_options, neuraldata
export save_model, θy0
export Sigmoid, Softplus
export noiseless_neuralDDM, θneural_noiseless, neural_options_noiseless
export neural_poly_DDM
export θneural_choice
export neural_choiceDDM, θneural_choice, neural_choice_options

export fit
export dimz
export likelihood, choice_loglikelihood, joint_loglikelihood
export choice_optimize, choice_neural_optimize, choice_likelihood
export simulate_expected_firing_rate, reload_neural_data
export loglikelihood, synthetic_data
export CIs, optimize, Hessian, gradient
export load_choice_data, reload_neural_model, save_neural_model, flatten
export save, load, reload_choice_model, save_choice_model
export reload_joint_model
export initialize_θy, neural_null
export synthetic_clicks, binLR, bin_clicks
export default_parameters_and_data, compute_LL
export mean_exp_rate_per_trial, mean_exp_rate_per_cond
export logprior, process_spike_data
export θprior, train_and_test, all_Softplus, θ2, invθ2, save_choice_data
export load_neural_data


include("types.jl")
include("choice_model/types.jl")
include("neural_model/types.jl")
include("neural-choice_model/types.jl")

include("base_model.jl")
include("analysis_functions.jl")
include("optim_funcs.jl")
include("sample_model.jl")
include("priors.jl")

include("choice_model/choice_model.jl")
include("choice_model/sample_model.jl")
include("choice_model/process_data.jl")

include("neural_model/neural_model.jl")
include("neural_model/sample_model.jl")
include("neural_model/process_data.jl")
include("neural_model/noiseless_model.jl")
include("neural_model/RBF_model.jl")

#include("neural_model/neural_model-sep.jl")
#include("neural_model/filter/filtered.jl")
#include("neural_model/neural_model-th.jl")

include("neural-choice_model/neural-choice_model.jl")
include("neural-choice_model/sample_model.jl")
include("neural-choice_model/neural-choice_model-ALT.jl")

#include("neural-choice_model/neural-choice_model-negBin.jl")
#include("neural-choice_model/process_data.jl")

##include("neural_model/null.jl")
##include("neural_model/polynomial/neural_poly_model.jl")
##include("neural_model/polynomial/noiseless_model_poly.jl")
##include("neural_model/load_and_optimize.jl")
##include("neural_model/sample_model_functions_FP.jl")

end
