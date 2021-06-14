"""
"""
@with_kw struct θprior{T<:Real} @deftype T
    μ_B = 5.
    σ_B = 60.
end


"""
    logprior(x, θprior)

"""
function logprior(x::Vector{T}, θprior::θprior) where {T <: Real}
    
    @unpack μ_B, σ_B = θprior
    
    # if (x[2] >= μ_B)
    #     logpdf(Laplace(μ_B, σ_B), x[2]) 
    # else
    #     logpdf(Laplace(μ_B, σ_B), μ_B)
    # end

    logpdf(InverseGamma(μ_B, σ_B), x[2])

end


"""
"""
function sigmoid_prior(x::Vector{T1}, θ::Union{θneural_noiseless, θneural}; 
        sig_σ::Float64=1.) where {T1 <: Real}

    @unpack f = θ
    θ = θneural_noiseless(x, f)
    
    if typeof(f) == String
        if f == "Sigmoid"
            sum(map(x-> sum(logpdf.(Normal(0., sig_σ), map(x-> x.c, x))), θ.θy))
        else
            0.
        end
    else    
        sum(map(x-> sum(logpdf.(Normal(0., sig_σ), x.c)), vcat(θ.θy...)[vcat(f...) .== "Sigmoid"]))
    end
    
end