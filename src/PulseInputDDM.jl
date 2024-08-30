"""
    Pulse Input DDM

A julia module for fitting bounded accumlator models using behavioral
and/or neural data from pulse-based evidence accumlation tasks.
"""

module PulseInputDDM

using DocStringExtensions
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

export load_choice_data, load_choice_model, save_choice_model, save_choice_data
export choiceDDM, θchoice
export neuralDDM, θneural
export θneural_choice, neural_choiceDDM
export fit
export load_neural_model, save_neural_model, load_neural_data
export initalize
export θz, θy
export loglikelihood
export CIs, Hessian, gradient
export synthetic_data
export all_Softplus
export Sigmoid, Softplus
export process_spike_data
export neural_choice_options, neural_options_noiseless, neural_options
export simulate_expected_firing_rate
export joint_loglikelihood, choice_loglikelihood
export choice_neural_optimize, choice_optimize

###

export neuraldata
export save_model
export noiseless_neuralDDM
export dimz
export likelihood
export choice_likelihood
export flatten
export reload_joint_model
export synthetic_clicks, binLR, bin_clicks
export default_parameters_and_data, compute_LL
export mean_exp_rate_per_trial, mean_exp_rate_per_cond

include("types.jl")
include("choice/types.jl")
include("neural/types.jl")
include("joint/types.jl")

include("core.jl")
include("utils.jl")
include("optimization.jl")
include("sampling.jl")

include("choice/choice.jl")
include("choice/sampling.jl")
include("choice/preprocessing.jl")
include("choice/IO.jl")

include("neural/neural.jl")
include("neural/sampling.jl")
include("neural/preprocessing.jl")
include("neural/initalize.jl")
include("neural/RBF_model.jl")
include("neural/IO.jl")

include("joint/joint.jl")
include("joint/sampling.jl")
include("joint/joint-ALT.jl")

end
