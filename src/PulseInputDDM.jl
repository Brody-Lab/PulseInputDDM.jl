"""
    Pulse Input DDM

A julia module for fitting bounded accumlator models using behavioral
and/or neural data from pulse-based evidence accumlation tasks.
"""

#__precompile__(false)

module PulseInputDDM

using StatsBase, Distributions, LineSearches, JLD2
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
export save_model

export Sigmoid, Softplus
export noiseless_neuralDDM, θneural_noiseless, neural_options_noiseless
export neural_poly_DDM
export θneural_choice
export neural_choiceDDM, θneural_choice, neural_choice_options
export θneural_choice_GLM, neural_choice_GLM_DDM, neural_choice_GLM_options

export dimz
export likelihood, choice_loglikelihood, joint_loglikelihood
export choice_optimize, choice_neural_optimize, choice_likelihood
export simulate_expected_firing_rate, reload_neural_data
export loglikelihood, synthetic_data
export CIs, optimize, Hessian, gradient
export load_choice_data, load_neural_data, reload_neural_model, save_neural_model, flatten
export save, load, reload_choice_model, save_choice_model
export reload_joint_model
export initialize_θy, neural_null
export synthetic_clicks, binLR, bin_clicks
export default_parameters_and_data, compute_LL
export mean_exp_rate_per_trial, mean_exp_rate_per_cond
export logprior, process_spike_data
export θprior, train_and_test, all_Softplus, θ2, invθ2, save_choice_data

abstract type DDM end
abstract type DDMdata end
abstract type DDMθ end
abstract type DDMf end


"""
    choiceDDM_dx(θ, data, dx, cross)

Fields:

- `θ`: a instance of the module-defined class `θchoice` that contains all of the model parameters for a `choiceDDM`
- `data`: an `array` where each entry is the module-defined class `choicedata`, which contains all of the data (inputs and choices).
- `dx`: width of spatial bin (defaults to 0.25).
- `cross`: whether or not to use cross click adaptation (defaults to false).
"""
@with_kw struct choiceDDM_dx{T,U,V} <: DDM
    θ::T = θchoice()
    data::U
    dx::Float64=0.25
    cross::Bool=false
    θprior::V = θprior()
end


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
include("neural_model/neural_model-sep.jl")
include("neural_model/sample_model.jl")
include("neural_model/process_data.jl")
include("neural_model/noiseless_model.jl")
#include("neural_model/null.jl")
#include("neural_model/polynomial/neural_poly_model.jl")
#include("neural_model/polynomial/noiseless_model_poly.jl")
include("neural_model/RBF_model.jl")
include("neural_model/filter/filtered.jl")
include("neural_model/neural_model-th.jl")

include("neural-choice_model/sample_model.jl")
include("neural-choice_model/neural-choice_model.jl")
include("neural-choice_model/neural-choice_model-negBin.jl")
include("neural-choice_model/neural-choice_model-sep.jl")
include("neural-choice_model/neural-choice_model-ALT.jl")
include("neural-choice_model/neural-choice_GLM_model.jl")
include("neural-choice_model/process_data.jl")

#include("neural_model/load_and_optimize.jl")
#include("neural_model/sample_model_functions_FP.jl")

end
