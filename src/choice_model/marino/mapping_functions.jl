#################################### Choice observation model #################################

function map_split_combine_marino(p_opt, p_const, fit_vec, dt,
    lb::Vector{Float64}, ub::Vector{Float64})
    
    pz, pd, pw = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec))
    pz = map_pz!(pz,dt,lb,ub)
    pd = map_pd!(pd)
    pw = map_pw!(pw)
    
    return pz, pd, pw
    
end

function split_combine_invmap_marino(pz::Vector{TT}, pd::Vector{TT}, pw::Vector{TT}, fit_vec, dt,
        lb::Vector{Float64}, ub::Vector{Float64}) where {TT <: Any}

    pz = inv_map_pz!(copy(pz),dt,lb,ub)
    pd = inv_map_pd!(copy(pd))
    pw = copy(pw)
    
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,pd,pw),fit_vec)

    return p_opt, p_const
    
end

split_latent_and_observation(p::Vector{TT}) where {TT} = p[1:dimz], p[dimz+1:dimz+2], p[dimz+3:end]

combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}}, 
    pd::Union{Vector{TT},BitArray{1}},
    pw::Union{Vector{TT},BitArray{1}}) where {TT} = vcat(pz,pd,pw)

map_pw!(x) = min.(max.(0.,x),1.)