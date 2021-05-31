"""
    θDDLM

A module-specific type that specifies the parameters of the joint model.

Fields:
- σ2_i: variance of the initial noise
- B: bound height
- λ: impulsiveness (λ>0) or leakiness (λ<0)
- σ2_a: variance of noise added each time bin
- σ2_s: variance of noise added to each click
- ϕ: adaptation or facilitation strength
- τ_ϕ: time constant of adaptation or facilitation
-`α`: The impact of the correct side of the previous trial
-`k`: The exponential change rate of α as a function of trial number in the past
-`bias`: a float that specifies the decision criterion across trials and trial-sets. At the end of each trial, the model chooses right if the integral of P(a) is greater than the bias
-`lapse`: a float indicating the fraction of trials when the animal makes a choice ignoring the accumulator value
"""
@with_kw struct θDDLM{T1<:θz, T2<:θh, T3<:Real} <: DDMθ
    θz::T1 = θz()
    θh::T2 = θh()
    bias::T3 = 0.
    lapse::T3 = 0.
end

"""
Constructor method for ([`θDDLM`](@ref)) from a vector of parameter values

Arguments:

- `x` The values of the model parameters
"""
function θDDLM(x::Vector{T}) where {T <: Real}
    θDDLM(θz(x[1:7]...), θh(x[8:9]...), x[10], x[11])
end

"""
    flatten(θ)

Convert an instance of [`θDDLM`](@ref) to a vector.

Arguments:

- `θ`: an instance of ['θDDLM'](@ref)

Returns:

- a vector of Floats
"""
function flatten(θ::θDDLM)

    @unpack θz, θh, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    @unpack α, k = θh
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, α, k, bias, lapse)
end

"""
    DDLMoptions

A module-specific type that specifies which parameters to fit, their lower and upper bounds, initial values, and other settings used during data preprocessing or model optimization.

Fields:

-`a_bases`: a vector of vector specifying the kernels with which the mean of the latent variable is filtered to construct regressors
-`centered`: Defaults to true. For the neural model, this aligns the center of the binned spikes, to the beginning of the binned clicks. This was done to fix a numerical problem. Most users will never need to adjust this.
-`cross`: Whether to implement cross-stream adaptation
-`datapath:` location of the MATLAB file containing the data and model specifications
-`dt`: Binning of the spikes, in seconds.
-`fit`: a BitArray indicating which parameters are fit
-`L2regularizer`: a square matrix specifying the L2-regularization penalty for each regressor. Different groups of regressors may have different penalty terms
-`lb`: a vector of floats indicating the lower bound of each parameter during optimization
-`n`: Number of bins in the space of the latent variable
-`nback`: number of trials in the past to consider
-`remap`: boolean indicating whether to compute in the space where the noise terms (σ2_a, σ2_i, σ2_s) are squared
-`resultspath`: where the results of the model fitting are to be saved
-`ub`: a vector of floats indicating the upper bound of each parameter during optimization
-`x0`: initial values of the parameters
"""
@with_kw struct DDLMoptions{T1<:BitArray, T2<:Vector{Float64}, T3<:Bool, T4<:String, T5<:AbstractFloat, T6<:Integer, T7<:Vector{Vector{Float64}}, T8<:Matrix{Float64}}
    a_bases::T7 = [ones(1)]
    centered::T3=true
    cross::T3=false
    datapath::T4=""
    dt::T5 = 1e-2
    fit::T1 = BitArray(undef,0)
    L2regularizer::T8=Matrix{Float64}(undef,0,0)
    lb::T2 = Vector{Float64}(undef,0)
    n::T6=53
    nback::T6=10
    npostpad_abar::T6=30
    remap::T3=false
    resultspath::T4=""
    ub::T2 = Vector{Float64}(undef,0)
    x0::T2 = Vector{Float64}(undef,0)
end

"""
    trialdata

Module-defined type containing information on the behavioral data in each trial

Fields:
- clickcounts: an instance of 'binned_clicks', containing information on the click count in each time bin
- clicktimes: an instance of 'clicks', containing information on the times of each click
- choice: 0 indicates a left choice, and 1 a right choice

"""
@with_kw struct trialdata{T1 <: binned_clicks, T2 <: clicks, T3 <: Bool}
    clickcounts::T1
    clicktimes::T2
    choice::T3
end

"""
    unitdata

Module-defined type containing design matrix and spike count of each unit

Fields:
- Xautoreg: component of the design matrix containing auroregressive regressors based on the spiking history
- y: column vector of spike count of the unit in each time bin in each trial, concatenated across trials

"""
@with_kw struct unitdata{T1 <: Matrix{Float64}, T2 <: Vector{Float64}}
    Xautoreg::T1
    y::T2
end

"""
    trialsetdata

Module-defined type containing information on the behavioral data in each trial and the spike count data of each unit

Fields:
- shifted: An instance of 'trialshifted'
- trials: A vector of 'trialdata' objects
- units: A vector of 'unitdata' objects
- Xtiming: the component of the design matrix that contains regressors related to the timing of events in each trial
"""
@with_kw struct trialsetdata{T1<:trialshifted, T2<:Vector, T3<:Vector, T4<:Matrix{Float64}}
    shifted::T1
    trials::T2
    units::T3
    Xtiming::T4
    @assert all(map(x->typeof(x)<:trialdata, trials))
    @assert all(map(x->typeof(x)<:unitdata, units))
end

"""
    DDLM

A module-specific type that instantiates a drift-diffusion linear model that is fitted to the choice and firing rates, using stimuli and trial history as inputs

Fields:
- data: an vector of 'trialsetdata'
- options: an instance of 'DDLMoptions'
- θ: an instance of ''θDDLM'
"""
@with_kw struct DDLM{T1<:Vector, T2<:DDLMoptions, T3<:θDDLM} <: DDM
    data::T1
    options::T2
    θ::T3
    @assert all(map(x->typeof(x)<:trialsetdata, data))
end
