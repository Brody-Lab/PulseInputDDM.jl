function map_split_combine_new(p_opt, p_const, fit_vec, dt,
    lb::Vector{Float64}, ub::Vector{Float64})
    
    pz, pd = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec))
    pz = map_pz!(pz,dt,lb,ub)
    pd = map_pd_new!(pd)
    
    return pz, pd
    
end

"""
    split_combine_invmap(pz, bias, fit_vec, dt)  

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap_new(pz::Vector{TT}, pd::Vector{TT}, fit_vec, dt,
        lb::Vector{Float64}, ub::Vector{Float64}) where {TT <: Any}

    pz = inv_map_pz!(copy(pz),dt,lb,ub)
    pd = inv_map_pd_new!(copy(pd))
    
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,pd),fit_vec)

    return p_opt, p_const
    
end

function map_pd_new!(x)
        
    x[1:2] = eps() .+ (1. - eps()) .* logistic.(x[1:2])
    x[3:4] = x[3:4] 
    
    return x
    
end

function inv_map_pd_new!(x)
    
    x[1:2] = logit.((x[1:2] .- eps()) .* inv.(1. - eps()))
    x[3:4] = x[3:4]
        
    return x
    
end