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
#using ImageFiltering
using ForwardDiff: value
using PositiveFactorizations
using Parameters, TransformVariables
import Base.rand
import Base.Iterators: partition
using StaticArrays, Flatten
import Flatten: flattenable

export choiceDDM, opt, θchoice, choicedata, θz
export θneural, neuralDDM, neuraldata, θy, neuraldata
export Sigmoid, Softplus

export dimz
export loglikelihood, synthetic_data
export CIs, optimize, Hessian, gradient
export load, reload, save, flatten

export default_parameters_and_data, compute_LL

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

abstract type DDM end
abstract type DDMdata end
abstract type DDMθ end

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
end


"""
"""
@with_kw struct neuralinputs{T1,T2,T3}
    clicks::T1
    binned_clicks::T2
    λ0::T3
    dt::Float64
    centered::Bool
end


"""
"""
@with_kw struct opt
    fit::Vector{Bool} = vcat(trues(9))
    lb::Vector{Float64} = vcat([0., 8., -5., 0., 0., 0.01, 0.005], [-30, 0.])
    ub::Vector{Float64} = vcat([2., 30., 5., 100., 2.5, 1.2, 1.], [30, 1.])
    x0::Vector{Float64} = vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01])
end


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
include("neural_model/deterministic_model.jl")

#include("neural_model/load_and_optimize.jl")
#include("neural_model/sample_model_functions_FP.jl")

end
