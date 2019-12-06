import Base.rand
import Distributions: logpdf

#ProjDir = @__DIR__
#cd(ProjDir)

#=
struct LNR{T1,T2,T3} <: ContinuousUnivariateDistribution
    μ::T1
    σ::T2
    ϕ::T3
end

#Broadcast.broadcastable(x::LNR) = Ref(x)

LNR(;μ, σ, ϕ) = LNR(μ, σ, ϕ)

function rand(dist::LNR)
    @unpack μ,σ,ϕ = dist
    x = @. rand(LogNormal(μ, σ)) + ϕ
    rt,resp = findmin(x)
    return resp,rt
end

rand(dist::LNR, N::Int) = [rand(dist) for i in 1:N]

function logpdf(d::T, r::Int, t::Float64) where {T<:LNR}
    @unpack μ,σ,ϕ = d
    LL = 0.0
    for (i,m) in enumerate(μ)
        if i == r
            LL += logpdf(LogNormal(m, σ), t-ϕ)
        else
            LL += log(1-cdf(LogNormal(m, σ), t-ϕ))
        end
    end
    return LL
end

logpdf(d::LNR, data::Tuple) = logpdf(d, data...)


struct choiceDDM{T1,T2,T3,T4,T5,T6,T7,T8,T9} <: ContinuousUnivariateDistribution
    σ2_i::T1
    B::T2
    λ::T3
    σ2_a::T4
    σ2_s::T5
    ϕ::T6
    τ_ϕ::T7
    bias::T8
    lapse::T9
end

function rand(dist::choiceDDM, L::Vector{Float64}, R::Vector{Float64}, T::Float64;
        dtMC::Float64=1e-4, rng::Int = 1, use_bin_center::Bool=false)
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse = dist
    pz = vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ)
    pd = vcat(bias, lapse)
    binned = pulse_input_DDM.bin_clicks(T,L,R; dt=dtMC, use_bin_center=use_bin_center)
    pulse_input_DDM.sample_choice_single_trial(L, R, binned..., pz, pd; use_bin_center=use_bin_center, dtMC=dtMC, rng=rng)
end

rand(dist::choiceDDM, N::Int) = [rand(dist) for i in 1:N]

=#
