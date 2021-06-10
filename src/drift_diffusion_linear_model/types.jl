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
-`coupling`: A nested vector whose element coupling[i][j] indicates the coupling between the j-th neuron in the i-th trialset and the latent variable
"""
@with_kw struct θDDLM{T<:Real, T2<:Vector{<:Vector{<:Real}}} <: DDMθ
    α::T = 0.
    B::T = 15.
    bias::T = 0.
    k::T = 0.
    λ::T = -0.5; @assert λ != 0.
    lapse::T = 0.
    ϕ::T = 0.8; @assert ϕ != 1.
    σ2_a::T = 50.
    σ2_i::T = 0.5; @assert σ2_i != 0.
    σ2_s::T = 1.5
    τ_ϕ::T = 0.05
    coupling::T2 = [[1.]]
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
    autoreg_bases::T8 = ones(1,1)
    centered::T3=true
    cross::T3=false
    datapath::T4=""
    dt::T5 = 1e-2
    fit::T1 = BitArray(undef,0)
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
    clickindices::T1
    clicktimes::T2
    choice::T3
end

"""
    unitdata

Module-defined type containing design matrix and spike count of each unit

Fields:
- L2Regularizer: a `p`-by-`p` matrix with nonzero diagonal terms to implment L2 regularization and to prevent having to invert a singular matrix
- ℓ₀y: column vector indicating the likelihood of the spike train conditioned on the latent being zero for all time bins and all trials
- X: the design matrix containing auroregressive regressors based on the spiking history and regressors related to the timing of trial events. The columns related to the latent variable variable are preallocated as zeros.
- y: column vector of spike count of the unit in each time bin in each trial, concatenated across trials
"""
@with_kw struct unitdata{T1 <: Matrix{Float64}, T2 <: Vector{Float64}}
    L2regularizer:: T1
    ℓ₀y::T2
    Xautoreg::T1
    y::T2
end

"""
    laggeddata

Module-defined type providing information on the lagged outcome and choices.

Fields:

- `choice`: number-of-trials by 2*number-of-lags+1 array of Int64. Each row  corresponds to a trial whose choice and firing ratees are being fitted, and each column corresponds to the choice a particular lag relative to that trial. A value of -1 indicates a left choice, 1 a right choice, and 0 no information at that lag. For example, `choice[i,j]' represents the choice lagged by `lag[j]` from trial i.
- `answer`: Same organization as `choice.` The values -1, 1, and 0 represent a previous left answer, a previous right answer, and lack of information on the answer on a trial in the past or future.
- `reward`: Same organization as `choice.` The values -1, 1, and 0 represent the absence of, presence of, and lack of information on reward on a trial in the past or future.
- `lag`: a row indicating the number of trials in the past (negative values) or future (positive values) represented by each column of `answer`, `choice`, and `reward`
- eˡᵃᵍ⁺¹: A row indicating the exponentiated values of the sum `lag`+1
"""
@with_kw struct laggeddata{T1<:Matrix{Int64}, T2<:Matrix{Float64}}
    answer::T1
    choice::T1
    eˡᵃᵍ⁺¹::T2
    lag::T1
    reward::T1
end

"""
    trialsetdata

Module-defined type containing information on the behavioral data in each trial and the spike count data of each unit

Fields:
- lagged: An instance of 'laggeddata'
- trials: A vector of 'trialdata' objects
- units: A vector of 'unitdata' objects
"""
@with_kw struct trialsetdata{T1<:laggeddata, T2<:Vector{<:Int}, T3<:Vector{<:trialdata}, T4<:Vector{<:unitdata}, T5<:Matrix{Float64}}
    lagged::T1
    nbins_each_trial::T2
    trials::T3
    units::T4
    Xtiming::T5
end

"""
Constructor method for ([`θDDLM`](@ref)) from a vector of parameter values

Arguments:

- `x` The values of the model parameters
- `nunits_each_trialset`: Number of neuronal units in each trialset
"""
function θDDLM(x::Vector{T}, data::Vector{<:trialsetdata}) where {T<:Real}
    fnames = collect(fieldnames(θDDLM))
    n_other_parameters = sum(fnames .!= :coupling)
    xcoupling = x[n_other_parameters+1:end]

    nunits_each_trialset = map(trialset->length(trialset.units), data)
    coupling = map(nunits->zeros(T, nunits), nunits_each_trialset)

    k = 0
    for i = 1:length(nunits_each_trialset)
        coupling[i] = view(xcoupling, (k+1):(k+=nunits_each_trialset[i]))
    end
    θDDLM(x[1:n_other_parameters]..., coupling)
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
    fnames = collect(fieldnames(θDDLM))
    fnames[fnames .!= :coupling]
    vcat(map(x->getfield(θ,x), fnames[fnames .!= :coupling]), vcat(θ.coupling...))
end

"""
    DDLM

A module-specific type that instantiates a drift-diffusion linear model that is fitted to the choice and firing rates, using stimuli and trial history as inputs

Fields:
- data: an vector of 'trialsetdata'
- options: an instance of 'DDLMoptions'
- θ: an instance of ''θDDLM'
"""
@with_kw struct DDLM{T1<:Vector{<:trialsetdata}, T2<:DDLMoptions, T3<:θDDLM} <: DDM
    data::T1
    options::T2
    θ::T3
end

"""
    latentspecification

A container of variable specifying the latent space
-cross: whether cross-stream adaptation is implemented
-dt: size of each time bin
-dx: size of the bins into which latent space is discretized
-M: P(aₜ|aₜ₋₁, θ, δₜ=0), i.e., a square matrix representing the transition matrix if no clicks occurred in the current time step
-n: number of bins into which latent space is discretized
-nprepad_abar: number of time bins to pad the beginning mean of the latent trajectory
-xc: centers of bins in latent space
"""
@with_kw struct latentspecification{T1<:Bool, T2<:Float64, T3<:Real, T4<:Matrix{<:Real}, T5<:Int, T6<:Vector{<:Real}}
    cross::T1
    dt::T2
    dx::T3
    M::T4
    n::T5
    nprepad_abar::T5
    npostpad_abar::T5
    xc::T6
end
