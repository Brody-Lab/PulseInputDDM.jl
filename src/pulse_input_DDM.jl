"""
    pulse_input_DDM

A julia module for fitting bounded accumlator models using behavioral
and/or neural data from pulse-based evidence accumlation tasks.
"""
module pulse_input_DDM

using StatsBase, Distributions, LineSearches, JLD2
using ForwardDiff, Distributed, LinearAlgebra
using Optim, DSP, SpecialFunctions, MAT, Random
using Discretizers
using ImageFiltering
using ForwardDiff: value
using PositiveFactorizations, Parameters, Flatten
using Polynomials, Missings
using HypothesisTests

import StatsFuns: logistic, logit, softplus, xlogy
import Base.rand
import Base.Iterators: partition
import Flatten: flattenable
#import Polynomials: Poly
using BasisFunctionExpansions

export choiceDDM, choiceoptions, θchoice, choicedata, θz
export θneural, neuralDDM, neuraldata, θy, neuraldata
export Sigmoid, Softplus, Sigmoid_options, Softplus_options

export θneural_filt, filtoptions, filtdata

export mixed_options_noiseless, θneural_noiseless_mixed, mixed_options, θneural_mixed
export θneural_th, th_options

#export neural_poly_DDM, θneural_poly

export Softplus_options_noiseless

export θneural_noiseless, Sigmoid_options_noiseless

export θneural_choice, Softplus_choice_options, neural_choice_data

export dimz
export loglikelihood, synthetic_data
export CIs, optimize, Hessian, gradient
export load, reload, save, flatten
export initialize_θy, neural_null
export synthetic_clicks, binLR, bin_clicks

export μ_poly_options

export default_parameters_and_data, compute_LL

export mean_exp_rate_per_trial, mean_exp_rate_per_cond
export logprior

#=

export compute_ΔLL

export choice_null
export sample_input_and_spikes_multiple_sessions, sample_inputs_and_spikes_single_session
export sample_spikes_single_session, sample_spikes_single_trial, sample_expected_rates_single_session

export sample_choices_all_trials
export aggregate_spiking_data, bin_clicks_spikes_and_λ0!

export diffLR

export filter_data_by_cell!

=#

abstract type DDM end
abstract type DDMdata end
abstract type DDMθ end
abstract type DDMf end

"""
"""
@with_kw struct θz{T<:Real} @deftype T
    σ2_i = 0.5
    B = 15.
    λ = -0.5; @assert λ != 0.
    σ2_a = 50.
    σ2_s = 1.5
    ϕ = 0.8; @assert ϕ != 1.
    τ_ϕ = 0.05
end


"""
"""
@with_kw struct clicks
    L::Vector{Float64}
    R::Vector{Float64}
    T::Float64
end


"""
"""
@with_kw struct binned_clicks
    #clicks::T
    nT::Int
    nL::Vector{Int}
    nR::Vector{Int}
end


@with_kw struct bins
    #clicks::T
    xc::Vector{Real}
    dx::Real
    n::Int
end


"""
"""
@with_kw struct choiceinputs{T1,T2}
    clicks::T1
    binned_clicks::T2
    dt::Float64
    centered::Bool
    delay::Int=0
    pad::Int=0
end

"""
"""
@with_kw struct neuralinputs{T1,T2}
    clicks::T1
    binned_clicks::T2
    λ0::Vector{Vector{Float64}}
    dt::Float64
    centered::Bool
    delay::Int
    pad::Int
end


"""
"""
neuralinputs(clicks, binned_clicks, λ0::Vector{Vector{Vector{Float64}}}, dt::Float64, centered::Bool, delay::Int, pad::Int) =
    neuralinputs.(clicks, binned_clicks, λ0, dt, centered, delay, pad)

include("base_model.jl")
include("analysis_functions.jl")
include("optim_funcs.jl")
include("sample_model.jl")

include("choice_model/choice_model.jl")
include("choice_model/compute_LL.jl")
include("choice_model/sample_model.jl")
include("choice_model/process_data.jl")

include("neural_model/neural_model.jl")
include("neural_model/compute_LL.jl")
include("neural_model/sample_model.jl")
include("neural_model/process_data.jl")
include("neural_model/noiseless_model.jl")
#include("neural_model/polynomial/neural_poly_model.jl")
#include("neural_model/polynomial/noiseless_model_poly.jl")
include("neural_model/filter/filtered.jl")
include("neural_model/neural_model-th.jl")

include("neural-choice_model/neural-choice_model.jl")
include("neural-choice_model/process_data.jl")

#include("neural_model/load_and_optimize.jl")
#include("neural_model/sample_model_functions_FP.jl")

end
