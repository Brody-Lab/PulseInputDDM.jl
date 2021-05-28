"""
    θDDLM

A module-specific type that specifies the parameters of the joint model.

Fields:
- `θz`: an instance of the module-specific type ['θz'](@ref) that contains the values of the parameters σ2_i, σ2_a, σ2_s, λ, B, ϕ, τ_ϕ
- `θh`: an instance of the module-specific type ['θh'](@ref) that contains the values parametrizing history-dependent influences
- `bias`: a float that specifies the decision criterion across trials and trial-sets. At the end of each trial, the model chooses right if the integral of P(a) is greater than the bias
- `lapse`: a float indicating the fraction of trials when the animal makes a choice ignoring the accumulator value
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
function θDDLM(x::Vector{T}) where {T <: AbstractFloat}
    θDDLM(θz=θz(σ2_i=x[1], B=x[2], λ=x[3], σ2_a=x[4], σ2_s=x[5], ϕ=x[6], τ_ϕ=x[7]),
          θh=θh(α=x[8],k=x[9]),
          bias=x[10], lapse=x[11])
end

"""
    vec(θ)

Convert an instance of [`θDDLM`](@ref) to a vector.

Arguments:

- `θ`: an instance of ['θDDLM'](@ref)

Returns:

- a vector of Floats
"""
function vec(θ::θDDLM)
    @unpack θz, θh, bias, lapse = θ
    vcat(map(x->getfield(θz, x), fieldnames(typeof(θz)))..., map(x->getfield(θh, x), fieldnames(typeof(θh)))..., bias, lapse)
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
@with_kw struct trialsetdata{T1<:trialshifted, T2, T3, T4<:Matrix{Float64}}
    shifted::T1
    trials::T2
    units::T3
    Xtiming::T4
end

"""
    DDLM

A module-specific type that instantiates a drift-diffusion linear model that is fitted to the choice and firing rates, using stimuli and trial history as inputs

Fields:
- data: an vector of 'trialsetdata'
- options: an instance of 'DDLMoptions'
- θ: an instance of ''θDDLM'
"""
@with_kw struct DDLM{T1<:θDDLM, T2<:Vector{trialsetdata}, T3<:DDLMoptions} <: DDM
    data::T2
    options::T3
    θ::T1
end
