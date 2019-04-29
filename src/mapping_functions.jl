#################################### Choice observation model #################################

"""
    map_split_combine(p_opt, p_const, fit_vec, dt)  

    Combine constant and variable optimization components, split into functional groups andmap to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt,
    lb::Vector{Float64}, ub::Vector{Float64})
    
    pz, pd = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec))
    pz = map_pz!(pz,dt,lb,ub)
    pd = map_pd!(pd)
    
    return pz, pd
    
end

"""
    split_combine_invmap(pz, bias, fit_vec, dt)  

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, pd::Vector{TT}, fit_vec, dt,
        lb::Vector{Float64}, ub::Vector{Float64}) where {TT <: Any}

    pz = inv_map_pz!(copy(pz),dt,lb,ub)
    pd = inv_map_pd!(copy(pd))
    
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,pd),fit_vec)

    return p_opt, p_const
    
end

split_latent_and_observation(p::Vector{TT}) where {TT} = p[1:dimz], p[dimz+1:dimz+2]
combine_latent_and_observation(pz::Vector{TT}, pd::Vector{TT}) where {TT} = vcat(pz,pd)

function map_pd!(x)
    
    lb = [-Inf, 0.]
    ub = [Inf, 1.]
    
    x[1] = x[1]       
    x[2] = lb[2] + (ub[2] - lb[2]) * normtanh(x[2])  
    
    return x
    
end

function inv_map_pd!(x)
    
    lb = [-Inf, 0.]
    ub = [Inf, 1.]
    
    x[1] = x[1]
    x[2] = normatanh((x[2] - lb[2])/(ub[2] - lb[2]))
        
    return x
    
end
    
    
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
      
#################### Common functions used in wrappers for all models ###########################
     
function map_pz!(x,dt,lb,ub)
    
    x[3] = lb[3] + (ub[3] - lb[3]) .* normtanh.(x[3])
    
    x[[1,2,4,5,6,7]] = lb[[1,2,4,5,6,7]] + exp.(x[[1,2,4,5,6,7]])
        
    return x
    
end

function inv_map_pz!(x,dt,lb,ub)

    x[3] = normatanh.((x[3] - lb[3])./(ub[3] - lb[3]))
    
    x[[1,2,4,5,6,7]] = log.(x[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]])
        
    return x
    
end
    
split_variable_and_const(p::Vector{TT},fit_vec::Union{BitArray{1},Vector{Bool}}) where TT = p[fit_vec],p[.!fit_vec]

function combine_variable_and_const(p_opt::Vector{TT}, p_const::Vector{Float64}, 
            fit_vec::Union{BitArray{1},Vector{Bool}}) where TT
    
    p = Vector{TT}(undef,length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    return p
    
end