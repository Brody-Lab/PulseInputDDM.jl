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


"""
"""
@with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end


"""
    neuralDDM

Fields:
- θ
- data
- n
- cross
- θprior

"""
@with_kw struct neuralDDM{T,U,V} <: DDM
    θ::T
    data::U
    n::Int=53
    cross::Bool=false
    θprior::V = θprior()
end


"""
"""
@with_kw struct θneural_noiseless{T1, T2} <: DDMθ
    θz::T1
    θy::T2
    f::Vector{Vector{String}}
end


"""
"""
@with_kw struct noiseless_neuralDDM{T,U} <: DDM
    θ::T
    data::U
end

