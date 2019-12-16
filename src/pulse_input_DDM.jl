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
using PositiveFactorizations
using Parameters, TransformVariables
import Base.rand

export dimz
export loglikelihood, synthetic_data
export CIs, optimize, Hessian, gradient
export load, reload, save

export mean_exp_rate_per_trial, mean_exp_rate_per_cond

export choiceDDM, opt, θchoice, choicedata, θz

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
@with_kw struct θchoice{T1, T<:Real}
    θz::T1 = θz()
    bias::T = 1.
    lapse::T = 0.05
end


"""
"""
@with_kw struct clicks
    L::Vector{Vector{Float64}}
    R::Vector{Vector{Float64}}
    T::Vector{Float64}
    ntrials::Int
end


"""
"""
@with_kw struct binned_clicks{T}
    clicks::T
    nT::Vector{Int}
    nL::Vector{Vector{Int}}
    nR::Vector{Vector{Int}}
    dt::Float64
    centered::Bool
end


"""
"""
@with_kw struct choicedata{T}
    binned_clicks::T
    choices::Vector{Bool}
end


"""
"""
@with_kw struct choiceDDM{T,U} <: ContinuousUnivariateDistribution
    θ::T = θchoice()
    data::U
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

end
