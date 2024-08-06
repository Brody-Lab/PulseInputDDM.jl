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
    choiceoptions(fit, lb, ub)

Fields:

- `fit`: `array` of `Bool` for optimization for `choiceDDM` model.
- `lb`: `array` of lower bounds for optimization for `choiceDDM` model.
- `ub`: `array` of upper bounds for optimization for `choiceDDM` model.
"""
@with_kw mutable struct choiceoptions
    fit::Vector{Bool} = vcat(trues(dimz+2))
    lb::Vector{Float64} = vcat([0., 4.,  -5., 0.,   0.,  0.01, 0.005], [-5, 0.])    
    ub::Vector{Float64} = vcat([30., 30., 5., 100., 2.5, 1.2,  1.], [5, 1.])
end


"""
    θchoice(θz, bias, lapse) <: DDMθ

Fields:

- `θz`: is a module-defined type that contains the parameters related to the latent variable model.
- `bias` is the choice bias parameter.
- `lapse` is the lapse parameter.

Example:

```julia
θchoice(θz=θz(σ2_i = 0.5, B = 15., λ = -0.5, σ2_a = 50., σ2_s = 1.5,
    ϕ = 0.8, τ_ϕ = 0.05), bias=1., lapse=0.05)
```
"""
@with_kw struct θchoice{T1, T<:Real} <: DDMθ
    θz::T1 = θz()
    bias::T = 1.
    lapse::T = 0.05
end


"""
    choicedata{T1} <: DDMdata

Fields:

- `click_data` is a type that contains all of the parameters related to click input.
- `choice` is the choice data for a single trial.

Example:

```julia
```
"""
@with_kw struct choicedata{T1} <: DDMdata
    click_data::T1
    choice::Bool
end


"""
    choiceDDM(θ, data, n, cross)

Fields:

- `θ`: a instance of the module-defined class `θchoice` that contains all of the model parameters for a `choiceDDM`
- `data`: an `array` where each entry is the module-defined class `choicedata`, which contains all of the data (inputs and choices).
- `n`: number of spatial bins to use (defaults to 53).
- `cross`: whether or not to use cross click adaptation (defaults to false).

Example:

```julia
ntrials, dt, centered, n  = 1, 1e-2, false, 53
θ = θchoice()
_, data = synthetic_data(n ;θ=θ, ntrials=ntrials, rng=1, dt=dt);
choiceDDM(θ=θ, data=data, n=n)
```
"""
@with_kw struct choiceDDM{T,U,V} <: DDM
    θ::T = θchoice()
    data::U
    n::Int=53
    cross::Bool=false
    θprior::V = θprior()
end
