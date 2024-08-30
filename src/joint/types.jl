"""
    neuralchoiceDDM

Fields:
- θ
- data
- n
- cross

"""
@with_kw mutable struct neural_choiceDDM{T} <: DDM
    θ::T
    n::Int=53
    cross::Bool=false
    fit::Vector{Bool}
    ub::Vector{Float64}
    lb::Vector{Float64}
end


"""
"""
@with_kw mutable struct θneural_choice{T1, T2, T3} <: DDMθ
    θz::T1
    bias::T2
    lapse::T2
    θy::T3
    f::Vector{Vector{String}}
end


"""
"""
function neural_choice_options(f)
    
    nparams, ncells = nθparams(f)
    fit = vcat(trues(dimz+2), trues.(nparams)...)
        
    lb = Vector(undef, sum(ncells))
    ub = Vector(undef, sum(ncells))
    
    for i in 1:sum(ncells)
        if vcat(f...)[i] == "Softplus"
            lb[i] = [-10]
            ub[i] = [10]
        elseif vcat(f...)[i] == "Sigmoid"
            lb[i] = [-100.,0.,-10.,-10.]
            ub[i] = [100.,100.,10.,10.]
        elseif vcat(f...)[i] == "Softplus_negbin"
            lb[i] = [0, -10]
            ub[i] = [Inf, 10]
        end
    end
    
    lb = vcat([1e-3, 8.,  -5., 1e-3,   1e-3,  1e-3, 0.005], [-10, 0.], vcat(lb...))
    ub = vcat([100., 40., 5., 400., 10., 1.2,  1.], [10, 1.], vcat(ub...));

    fit, lb, ub
    
end
   

"""
"""
function θneural_choice(x::Vector{T}, f::Vector{Vector{String}}) where {T <: Real}
    
    nparams, ncells = nθparams(f)
    
    borg = vcat(dimz + 2,dimz + 2 .+cumsum(nparams))
    blah = [x[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    blah = map((f,x) -> f(x...), getfield.(Ref(@__MODULE__), Symbol.(vcat(f...))), blah)
    
    borg = vcat(0,cumsum(ncells))
    θy = [blah[i] for i in [borg[i-1]+1:borg[i] for i in 2:length(borg)]]
    
    θneural_choice(θz(x[1:dimz]...), x[dimz+1], x[dimz+2], θy, f)

end


"""
    flatten(θ)

Extract parameters related to a `neural_choiceDDM` from an instance of `θneural_choice` and returns an ordered vector.
```
"""
function flatten(θ::θneural_choice)

    @unpack θy, θz, bias, lapse = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, bias, lapse,
        vcat(collect.(Flatten.flatten.(vcat(θy...)))...))

end