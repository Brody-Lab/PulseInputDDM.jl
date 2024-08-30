abstract type DDM end
abstract type DDMdata end
abstract type DDMθ end
abstract type DDMf end

"""
"""
@with_kw mutable struct θz{T<:Real} @deftype T
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