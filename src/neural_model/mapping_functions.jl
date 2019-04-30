    
#################################### Poisson neural observation model #########################

"""
    map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String)  

    Combine constant and variable optimization components, split into functional groups and map to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, N::Int, dimy::Int,
    lb::Vector{Float64}, ub::Vector{Float64})

    pz,py = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec), N, dimy)
    pz = map_pz!(pz,dt,lb,ub)       
    py = map_py!.(py,f_str=f_str)
    
    return pz,py
    
end

"""
    split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, fit_vec, dt, f_str::String)

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, fit_vec, dt, f_str::String, 
        lb::Vector{Float64}, ub::Vector{Float64}) where {TT <: Any}

    pz = inv_map_pz!(copy(pz), dt, lb, ub)     
    py = inv_map_py!.(deepcopy(py), f_str=f_str)
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,py),fit_vec)
    
    return p_opt, p_const
    
end

function split_latent_and_observation(p::Vector{T}, N::Int, dimy::Int) where {T <: Any}
                
    pz = p[1:dimz]
    py = reshape(p[dimz+1:dimz+dimy*N],dimy,N)
    py = map(i->py[:,i],1:N)

    return pz, py
    
end

combine_latent_and_observation(pz,py) = vcat(pz,vcat(py...))
    
function map_py!(p::Vector{TT};f_str::String="softplus") where {TT}
        
    if f_str == "exp"
        
        p[1] = exp(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        p[1:2] = exp.(p[1:2])
        p[3:4] = p[3:4]
        
    elseif f_str == "sig2"

        p[3:4] = p[3:4]
        p[1] = p[1]
        p[2] = exp(p[2])
        
    elseif f_str == "softplus"
          
        p[1] = exp(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function inv_map_py!(p::Vector{TT};f_str::String="softplus") where {TT}
     
    if f_str == "exp"

        p[1] = log(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        p[1:2] = log.(p[1:2])
        p[3:4] = p[3:4]
        
    elseif f_str == "sig2"

        p[1] = p[1]
        p[2] = log(p[2])
        p[3:4] = p[3:4]
        
    elseif f_str == "softplus"
        
        p[1] = log(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end
