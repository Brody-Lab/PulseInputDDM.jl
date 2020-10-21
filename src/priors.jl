"""
"""
@with_kw struct θprior{T<:Real} @deftype T
    μ_B = 40.
    σ_B = 1e4
end


"""
"""
function logprior(x,μ,σ) 
    
    if (x[2] >= μ[2])
        logpdf(Laplace(μ[2], σ[2]), x[2]) 
    else
        logpdf(Laplace(μ[2], σ[2]), μ[2])
    end

end