"""
A module-specific type that specifies the parameters of the joint model that are related to trial history

Fields:

-`α`: The impact of the correct side of the previous trial
-`k`: The exponential change rate of α as a function of trial number in the past
"""
@with_kw struct θh{T<:Real} @deftype T
    α = 0.
    k = 0.
end

"""
    θjoint

A module-specific type that specifies the parameters of the joint model.

Fields:
- `θz`: an instance of the module-specific type ['θz'](@ref) that contains the values of the parameters σ2_i, σ2_a, σ2_s, λ, B, ϕ, τ_ϕ
- `θh`: an instance of the module-specific type ['θz'](@ref) that contains the values parametrizing history-dependent influences
- `bias`: a float that specifies the decision criterion across trials and trial-sets. At the end of each trial, the model chooses right if the integral of P(a) is greater than the bias
- `lapse`: a float indicating the fraction of trials when the animal makes a choice ignoring the accumulator value
- `θy`: An instance of the module-specific type [`θy`]()@ref) that parametrizes the relationship between firing rate and the latent variable, a.
- `f`: A vector of vector of strings that
"""
@with_kw struct θjoint{T1<:θz, T2<:θh, T3<:Real, T4, T5<:Vector{Vector{String}}} <: DDMθ
    θz::T1 = θz()
    θh::T2 = θh()
    bias::T3 = 0.
    lapse::T3 = 0.
    θy::T4
    f::T5
end

"""
    trialsequence

Module-defined type providing information on the entire sequence of trials within which all the trials from each array of `neuraldata` is embedded.

Fields:
- `choice`: true if a right, and false if leftn
- `ignore`: true if the trial should be ignored for various reasons, such as breaking fixation or responding after too long of a delay
- `index`: the temporal position within the entire trial sequence of each trial whose choice and neural activity are being fitted.
- `reward`: true if rewarded
- `sessionstart`: true if first trial of a daily session
"""
@with_kw struct trialsequence{T1<:Vector{Bool}, T2<:Vector{Int64}}
    choice::T1
    ignore::T1
    index::T2
    reward::T1
    sessionstart::T1
end

"""
    trialshifted

Module-defined type providing information on the trial-shifted outcome and choices.

Fields:

- `choice`: number-of-trials by 2*number-of-shifts+1 array of Int. Each row  corresponds to a trial whose choice and firing ratees are being fitted, and each column corresponds to the choice a particular shift relative to that trial. A value of -1 indicates a left choice, 1 a right choice, and 0 no information at that shift. For example, `choice[i,j]' represents the choice shifted by `shift[j]` from trial i.
- `reward`: Same organization as `choice.` The values -1, 1, and 0 represent the absence of, presence of, and lack of information on reward on a trial in the past or future.
- `shift`: a vector indicating the number of trials shifted in the past (negative values) or future (positive values) represented by each column of `choice` and `reward`
"""
@with_kw struct trialshifted{T1<:Matrix{Int64}, T2>:Matrix{Int64}}
    choice::T1
    reward::T1
    shift::T2
end

"""
A module-specific type that specifies which parameters to fit and their lower and upper bounds and initial values

Fields:

-`fit` a vector of Bool indicating which parameters are fit
-`ub` a vector of floats indicating the upper bound of each parameter during optimization
-`lb` a vector of floats indicating the lower bound of each parameter during optimization
-`n` number of bins in which the latent space, a, is discretized
-`cross` whether to adapt clicks across left and right streams, as opposed to within each stream
"""
@with_kw struct joint_options{T1<:BitArray, T2<:Vector{Float64}}
    fit::T1
    ub::T2
    lb::T2
end

"""
    jointdata

Module-defined type containing information on the behavioral and neural data on each trial to be fitted, as well as the overall trial sequence that contains the subset of trials to be fitted.

Fields:
- An array of 'neuraldata'. Each array has the same neurons recorded in each trial
- trialsequence (['trialsequence'](@ref))

"""
@with_kw struct jointdata{T1<: Vector{neuraldata}, T2<:trialsequence, T3<:trialshifted} <: DDMdata
    neural_data::T1
    sequence::T2
    shifted::T3
end

"""
    jointDDM

A module-specific type that instantiates a drift-diffusion model that is fitted to the choice and firing rates, using stimuli and trial history as inputs

Fields:
- θ: a vector of
- [`jointdata`](@ref)
- `n`: number of bins in the space of the latent variable (a)
- cross: adaptation of sound pulses is cross-stream if true and within-stream otherwise
"""
@with_kw struct jointDDM{T1<:θjoint, T2<:Vector{}, T3<:Int64, T4<:Bool} <: DDM
    θ::T1
    joint_data::T2
    n::T3=53
    cross::T4=false
end

"""
    *** This is not currently used***

    settings

Additional parameters for processing data the data and fitting the model.

Fields:

- `break_sim_data`: this will break up simulatenously recorded neurons, as if they were recorded independently. Not often used by most users.
- `centered`: Defaults to true. For the neural model, this aligns the center of the binned spikes, to the beginning of the binned clicks. This was done to fix a numerical problem. Most users will never need to adjust this.
- `cut`: How much extra to cut off at the beginning and end of filtered things (should be equal to `extra_pad` in most cases).
- `delay`: How much to offset the spikes, relative to the accumlator, in units of `dt`.
- `dt`: Binning of the spikes, in seconds.
- `extra_pad`: Extra padding (in addition to `pad`) to add, for filtering purposes. In units of `dt`.
- `filtSD`: standard deviation of a Gaussin (in units of `dt`) to filter the spikes with to generate single trial firing rates (`μ_rnt`), and mean firing rate across all trials (`μ_t`).
- `pad`: How much extra time should spikes be considered before and after the begining of the clicks. Useful especially if delay is large.
- `pcut`: p-value for selecting cells.
"""
@with_kw struct settings{T1,T2,T3}
    break_sim_data::T1=false
    centered::T1=true
    cut::T2=10
    delay::T2=0
    do_RBF::T1=false
    dt::T3=1e-2
    extra_pad::T2=10
    filtSD::T2=2
    nRBFs::T2=6
    pad::T2=0
    pcut::T3=0.01
    @assert T1 == Bool
    @assert T2 <: Integer
    @assert T3 <: AbstractFloat
end
