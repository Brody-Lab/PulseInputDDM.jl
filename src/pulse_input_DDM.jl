"""
    pulse_input_DDM

A julia module for fitting bounded accumlator models using behavioral and/or neural data from pulse-based evidence accumlation tasks.
"""
module pulse_input_DDM

using StatsBase, Distributions, LineSearches, JLD2
using ForwardDiff, Distributed, LinearAlgebra
using Optim, DSP
using SpecialFunctions, MAT, Random
using Discretizers, PositiveFactorizations
using ImageFiltering
using ForwardDiff: value
using Parameters, Flatten
import Base.rand
import Base.Iterators: partition
import Flatten: flattenable
import StatsFuns: logistic, logit, softplus, xlogy

"""
paramdefs

"""
abstract type DDM end
abstract type DDMdata end
abstract type DDMθ end
abstract type θz_ch end
abstract type θz_baseprm end
abstract type DDMθoptions end


@with_kw struct θz_base{T<:Real} <: θz_baseprm @deftype T
    Bm = 0.; @assert Bm == 0.
    Bλ = 0.; @assert Bλ == 0.
    B0 = 1.8
    λ = -0.01; @assert λ != 0.
    σ2_i = eps()
    σ2_a = 0.
    σ2_s = 2.
    ϕ = 1.; #@assert ϕ != 1.
    τ_ϕ = 0.05
    bias = 0.
    lapse = 0.
    lapse_u = 0.
    h_drift_scale = 0.  
end


@with_kw struct θz_base_mod{T<:Real} <: θz_baseprm @deftype T
    Bm = 0.; @assert Bm == 0.
    Bλ = 0.; @assert Bλ == 0.
    B0 = 1.8
    λ = -0.01; @assert λ != 0.
    σ2_i = eps()
    σ2_a = 0.
    σ2_s = 2.
    ϕ = 1.; #@assert ϕ != 1.
    τ_ϕ = 0.05
    bias = 0.
    lapse = 0.
    lapse_u = 0.
    lapse_sig = 0.
    h_drift_scale = 0.  
end

@with_kw struct θz_expfilter{T<:Real} @deftype T
    h_η = 0.3
    h_β = 0.1929
end

@with_kw struct θz_expfilter_ce{T<:Real} <: θz_ch @deftype T 
    h_ηC = 0.3012
    h_ηE = 0.3012
    h_βC = 0.1929
    h_βE = 0.1929
end

@with_kw struct θz_expfilter_ce_bias{T<:Real} <: θz_ch @deftype T
    h_ηC = 0.3012
    h_ηE = 0.3012
    h_βC = 0.1929
    h_βE = 0.1929
    h_Cb = 0.
    h_Eb = 0.
end

@with_kw struct θz_expfilter_ce_lr{T<:Real} <: θz_ch @deftype T
    h_ηcr = 0.3012
    h_ηcl = 0.3012
    h_ηer = 0.3012
    h_ηel = 0.3012
    h_βcr = 0.1929
    h_βcl = 0.1929
    h_βer = 0.1929
    h_βel = 0.1929
end

@with_kw struct θz_expfilter_ce_lr_red{T<:Real} <: θz_ch @deftype T
    h_ηcr = 0.3012
    h_ηcl = 0.3012
    h_ηer = 0.3012
    h_ηel = 0.3012
    h_βc  = 0.1929
    h_βe  = 0.1929
end

@with_kw struct θz_DBM{T<:Real} @deftype T
    h_α = 0.7
    h_u = 0.5
    h_v = 2.
end

@with_kw struct θz_DBMexp{T<:Real} @deftype T
    h_α = 0.7
    h_u = 0.5
    h_v = 2.
end

@with_kw struct θz_DBMexp_sticky{T<:Real} <: θz_ch @deftype T
    h_α = 0.7
    h_u = 0.5
    h_v = 2.
    h_βc = 0.05 # choice stickiness 
end

@with_kw struct θz_LPSexp{T<:Real} @deftype T
    h_α = 0.8
    h_β = 0.05
    h_C = 0.05
end

@with_kw struct θz_Qlearn{T<:Real} <: θz_ch @deftype T
    h_αr = 0.1
    h_αf = 0.01
    h_κlc = .5
    h_κle = .05
    h_κrc = .5
    h_κre = .05
end

@with_kw struct θz_DBMexp_Qlearn{T<:Real} <: θz_ch @deftype T
    h_α = 0.7
    h_u = 0.5
    h_v = 2.
    h_αr = 0.1
    h_αf = 0.01
    h_κlc = .5
    h_κle = .05
    h_κrc = .5
    h_κre = .05
end

@with_kw struct θz_ndtime{T<:Real} @deftype T
    ndtimeL1 = 0.2
    ndtimeL2 = 0.03
    ndtimeR1 = 0.03
    ndtimeR2 = 0.02
end

@with_kw struct θz_ndtime_mod{T<:Real} @deftype T
    nd_θL = 0.1
    nd_θR = 0.1
    nd_vL = 10.
    nd_vR = 10.
    nd_tmod = 0.
    nd_vE = 0.
end    

@with_kw struct θ_expfilter <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_expfilter() 
    lpost_space::Bool = false 
end

@with_kw struct θ_expfilter_ce <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_expfilter_ce()
    lpost_space::Bool = false
end

@with_kw struct θ_expfilter_ce_bias <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_expfilter_ce_bias()
    lpost_space::Bool = false
end


@with_kw struct θ_expfilter_ce_lr <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_expfilter_ce_lr()
    lpost_space::Bool = false
end

@with_kw struct θ_expfilter_ce_lr_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_expfilter_ce_lr()
    lpost_space::Bool = false
end

@with_kw struct θ_expfilter_ce_lr_ndmod_lapsemod <: DDMθ
    base_θz = θz_base_mod()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_expfilter_ce_lr()
    lpost_space::Bool = false
end

@with_kw struct θ_expfilter_ce_lr_red_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_expfilter_ce_lr_red()
    lpost_space::Bool = false
end

@with_kw struct θ_expfilter_ce_lr_red_ndmod_lapsemod <: DDMθ
    base_θz = θz_base_mod()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_expfilter_ce_lr_red()
    lpost_space::Bool = false
end


@with_kw struct θ_DBM <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_DBM()
    lpost_space::Bool = true
end

@with_kw struct θ_DBMexp <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_DBMexp()
    lpost_space::Bool = true
end

@with_kw struct θ_DBMexp_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_DBMexp()
    lpost_space::Bool = true
end

@with_kw struct θ_DBMexp_sticky_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_DBMexp_sticky()
    lpost_space::Bool = true
end


@with_kw struct θ_LPSexp <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_LPSexp()
    lpost_space::Bool = true
end

@with_kw struct θ_Qlearn <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_Qlearn()
    lpost_space::Bool = true
end

@with_kw struct θ_Qlearn_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_Qlearn()
    lpost_space::Bool = true
end

@with_kw struct θ_DBMexp_Qlearn <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime()
    hist_θz = θz_DBMexp_Qlearn()
    lpost_space::Bool = true
end

@with_kw struct θ_DBMexp_Qlearn_ndmod <: DDMθ
    base_θz = θz_base()
    ndtime_θz = θz_ndtime_mod()
    hist_θz = θz_DBMexp_Qlearn()
    lpost_space::Bool = true
end

@with_kw struct choiceoptions <: DDMθoptions
	lb::Vector{Float64}
	ub::Vector{Float64}
	x0::Vector{Float64}
	fit::Vector{Bool}
end

@with_kw mutable struct clicks
	L::Vector{Float64}
	R::Vector{Float64}
	T::Float64
	gamma::Float64
end

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

@with_kw struct choiceinputs{T1,T2}
    clicks::T1
    binned_clicks::T2
    dt::Float64
    centered::Bool
end

@with_kw struct choicedata{T1} <: DDMdata
    click_data::T1
    choice::Bool
    sessbnd::Int64
end

@with_kw struct choiceDDM{T,U} <: DDM
    θ::T 
    data::U
end

export choiceDDM, choicedata, choiceoptions, choiceinputs
export θ_expfilter, θ_expfilter_ce, θz_base, θz_base_mod, θz_ndtime
export θz_expfilter, θz_expfilter_ce, θ_LPSexp, θ_DBMexpbnd, θ_DBMexp_ndmod
export θz_DBM, θz_DBMexp, θ_DBM, θ_DBMexp, θz_LPSexp
export θz_Qlearn, θ_Qlearn, θ_Qlearn_ndmod 
export θz_expfilter_ce_bias, θ_expfilter_ce_bias 
export θ_expfilter_ce_lr, θ_expfilter_ce_lr_ndmod, θz_expfilter_ce_lr
export θ_expfilter_ce_lr_red_ndmod_lapsemod, θ_expfilter_ce_lr_ndmod_lapsemod 
export θz_ndtime_mod, θz_ch
export θz_DBMexp_Qlearn, θ_DBMexp_Qlearn, θ_DBMexp_Qlearn_ndmod
export θz_expfilter_ce_lr_red, θ_expfilter_ce_lr_red_ndmod
export θz_DBMexp_sticky, θ_DBMexp_sticky_ndmod


const modeldict = Dict("expfilter" => θ_expfilter,
					"expfilter_ce" => θ_expfilter_ce,
                    "expfilter_ce_bias" => θ_expfilter_ce_bias,
                    "expfilter_ce_lr" => θ_expfilter_ce_lr,
                    "expfilter_ce_lr_ndmod" => θ_expfilter_ce_lr_ndmod,
                    "expfilter_ce_lr_ndmod_lapsemod" => θ_expfilter_ce_lr_ndmod_lapsemod,
                    "expfilter_ce_lr_red_ndmod" => θ_expfilter_ce_lr_red_ndmod,
                    "expfilter_ce_lr_red_ndmod_lapsemod" => θ_expfilter_ce_lr_red_ndmod_lapsemod,
                    "DBM"          => θ_DBM,
                    "DBMexp"       => θ_DBMexp,
                    "DBMexp_ndmod" => θ_DBMexp_ndmod,
                    "DBMexp_sticky_ndmod" => θ_DBMexp_sticky_ndmod,
                    "LPSexp"       => θ_LPSexp,
                    "Qlearn"       => θ_Qlearn,
                    "Qlearn_ndmod" => θ_Qlearn_ndmod,
                    "DBMexp_Qlearn" => θ_DBMexp_Qlearn,
                    "DBMexp_Qlearn_ndmod" => θ_DBMexp_Qlearn_ndmod)


"""
"""

function create_θ(modeltype)
	if modeltype in keys(modeldict)
		options, params = create_options(modeldict[modeltype]())
		return modeldict[modeltype](), options, params
	else
		error("Unknown model identifier $modeltype")
	end
end

"""
"""

function create_options(θ::DDMθ)

	 paramlims = Dict(  #:paramname => [lb, ub, fit, initialvalue]
    	:Bm => [0., 10., 0, 0.], :Bλ => [-5.0, 1.0, 0, 0.], :B0 => [0.5, 8.0, 1, 2.0],  	# bound parameters
    	:λ => [-30.0, 30.0, 1, 5.],                           					# leak
    	:σ2_i => [0.0, 2.0, 0, eps()], :σ2_a => [0.0, 10., 0, eps()], :σ2_s => [0.0, 20., 1, 2.],  # noise params
    	:ϕ => [0.01, 1.2, 1, 0.2], :τ_ϕ => [0.005, 1.0, 1, 0.02],        	# adaptation params
    	:bias => [-1.5, 1.5, 0, 0.],                                        # bias
        :lapse => [0.0, 0.5, 1, 1e-2], :lapse_u => [0.0, 0.8, 1, 0.02],         # lapse prob, mean params
        :lapse_sig => [0.0, 10., 1, 0.05],                                  # lapse sigmoidal modulation
    	:h_drift_scale => [0.0, 1.0, 1, 0.],                       # history drift scale    
        :lpost_space => [0 1 0 0],                                          # NOT A REAL VARIABLE - specifies whether model runs in logpost space

    	:ndtimeL1 => [0.0, 10.0, 1, 3.], :ndtimeL2 => [0.0, 5.0, 1, 0.04],  # ndtime left choice
    	:ndtimeR1 => [0.0, 10.0, 1, 3.], :ndtimeR2 => [0.0, 5.0, 1, 0.04],  # ndtime right choice
        :nd_θR => [0., 10., 1, 0.4], :nd_θL => [0., 10., 1, 0.4],               # ndtime mod bounds
        :nd_vL => [0., 12., 1, 4.], :nd_vR => [0., 12., 1, 4.],               # ndtime C mod drifts
        :nd_tmod => [0., 0.1, 1, 1e-3], :nd_vE => [-12., 12., 1, -2.],  # ndtime mod trial, and E mod drifts
    	
        :h_η => [-2.0, 2.0, 1, 0.3], :h_β => [0., 1., 1, 0.1],				# expfilter params
    	:h_ηC => [-2.0, 2.0, 1, 0.3], :h_ηE => [-2., 2., 1, 0.3],			# expfilter_ce params
    	:h_βC => [0., 1., 1, 0.1], :h_βE => [0., 1., 1, 0.1],				# expfilter_ce params
        :h_Cb => [-1., 1., 1, 0.], :h_Eb => [-1., 1., 1, 0.],             # expfilter_ce_bias params 
        :h_α => [0., 1., 1, 0.8], :h_u => [0., 1., 1, 0.5], :h_v => [0., 20., 1, 2.],    # DBM, DBMexp params
        :h_C => [0., 1., 1, 0.05],                                           # LPSexp along with h_α, h_β                                   
        :h_αr => [0., 1., 1, 0.9], :h_αf => [0., 1., 1, 0.1],               # Qlearning remember, forgetting rates
        :h_κlc => [0., 1., 1, 0.5], :h_κle => [0., 1., 1, 0.05],              # Qlearning left prediction errors : correct, error  
        :h_κrc => [0., 1., 1, 0.5], :h_κre => [0., 1., 1, 0.05],              # Q learning right prediction errors : correct, error 
        :h_ηcr => [-2.5, 2.5, 1, 0.3], :h_ηcl => [-2.5, 2.5, 1, 0.3],           # expfilter_ce_lr params
        :h_ηer => [-2.5, 2.5, 1, -0.3], :h_ηel => [-2.5, 2.5, 1, 0.3],           # expfilter_ce_lr params
        :h_βcr => [0., 1., 1, 0.1], :h_βcl => [0., 1., 1, 0.1],               # expfilter_ce_lr params
        :h_βer => [0., 1., 1, 0.1], :h_βel => [0., 1., 1, 0.1],               # expfilter_ce_lr params
        :h_βc => [0., 1., 1, 0.1], :h_βe => [0., 1., 1, 0.1])                 # expfilter_ce_lr_red params

	params = get_param_names(θ)

	ub = Array{Float64,1}(undef,length(params))
	lb = Array{Float64,1}(undef,length(params))
	fit = Array{Bool,1}(undef,length(params))
	x0 = Array{Float64,1}(undef,length(params))

	for i in 1:length(params)
    	lb[i] = paramlims[params[i]][1]
    	ub[i] = paramlims[params[i]][2]
    	fit[i] = paramlims[params[i]][3]
    	# x0[i] = lb[i] + (ub[i] - lb[i]) * rand()
    	x0[i] = paramlims[params[i]][4]
	end
	# x0[fit .== false] .= 0.
	options = choiceoptions(lb = lb, ub = ub, x0 = x0, fit = fit)
	return options, params

end



include("base_model.jl")
include("sample_model.jl")
include("helper_functions.jl")

include("choice_model/compute_LL.jl")
include("choice_model/process_data.jl")
include("choice_model/choice_optim.jl")
include("choice_model/compute_initial_pt.jl")


export create_θ, create_options, get_param_names, reconstruct_model
export loglikelihood, synthetic_data, synthetic_clicks
export CIs, optimize, Hessian, gradient, objectivefn
export load, reload, save, flatten, unflatten
export synthetic_clicks, binLR, bin_clicks
export compute_initial_pt, compute_bnd


end
