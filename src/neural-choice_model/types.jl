"""
    neuralchoiceDDM

Fields:
- θ
- data
- n
- cross

"""
@with_kw struct neural_choiceDDM{T} <: DDM
    θ::T
    n::Int=53
    cross::Bool=false
end


"""
"""
@with_kw struct θneural_choice{T1, T2, T3} <: DDMθ
    θz::T1
    bias::T2
    lapse::T2
    θy::T3
    f::Vector{Vector{String}}
end