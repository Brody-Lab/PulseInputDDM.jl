"""
    neuralchoiceDDM

Fields:
- θ
- data
- n
- cross

"""
@with_kw struct neural_choiceDDM{T,U} <: DDM
    θ::T
    data::U
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


@with_kw struct neural_choice_GLM_DDM{T,U} <: DDM
    θ::T
    data::U
    n::Int=53
    cross::Bool=false
end


"""
"""
@with_kw struct θneural_choice_GLM{T1, T2, T3} <: DDMθ
    stim::T1
    bias::T2
    lapse::T2
    θy::T3
    f::Vector{Vector{String}}
end