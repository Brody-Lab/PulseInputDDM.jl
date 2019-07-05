    
#################################### Poisson neural observation model #########################

"""
    map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String)  

    Combine constant and variable optimization components, split into functional groups and map to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, N::Vector{Int}, dimy::Int,
    lb::Vector{Float64}, ub::Vector{Float64})

    pz,py = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec), N, dimy)
    pz = map_pz!(pz,dt,lb,ub)       
    py = map(py-> map_py!.(py,f_str), py)
    
    return pz,py
    
end

function split_latent_and_observation(p::Vector{T}, N::Vector{Int}, dimy::Int) where {T <: Any}
                
    pz = p[1:dimz]
    #linear index that defines the beginning of a session
    iter = cumsum(vcat(0,N))*dimy
    #group single session parameters into 2D arrays
    py = map(i-> reshape(p[dimz+iter[i-1]+1:dimz+iter[i]], dimy, N[i-1]), 2:length(iter))
    #break up single session 2D arrays into an array of arrays
    py = map(i-> map(j-> py[i][:,j], 1:N[i]), 1:length(N))

    return pz, py
    
end

"""
    split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, fit_vec, dt, f_str::String)

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{Vector{TT}}}, fit_vec::BitArray{1}, dt, f_str::String, 
        lb::Vector{Float64}, ub::Vector{Float64}) where {TT <: Any}

    pz = inv_map_pz!(copy(pz), dt, lb, ub)     
    py = map(py-> inv_map_py!.(py, f_str), deepcopy(py))
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,py),fit_vec)
    
    return p_opt, p_const
    
end

combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}}, 
    py::Union{Vector{Vector{Vector{TT}}},Vector{Vector{BitArray{1}}}}) where {TT <: Any} = vcat(pz,vcat(vcat(py...)...))
    
function map_py!(p::Vector{TT}, f_str::String) where {TT}
    
    #eventually make this an input
    lb = [eps(),eps(),-Inf,-Inf]
    ub = [100., 100., Inf, Inf]
        
    if f_str == "exp"
        
        p[1] = exp(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        #p[1:2] = exp.(p[1:2])
        #p[1:2] = lb[1:2] .+ (ub[1:2] .- lb[1:2]) .* logistic.(p[1:2])
        p[1:2] = lb[1:2] .+ (ub[1:2] .- lb[1:2]) .* logistic!.(p[1:2])
        p[3:4] = p[3:4]
        
    elseif f_str == "softplus"
          
        p[1] = exp(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end

function inv_map_py!(p::Vector{TT}, f_str::String) where {TT}
    
    lb = [eps(),eps(),-Inf,-Inf]
    ub = [100., 100., Inf, Inf]
     
    if f_str == "exp"

        p[1] = log(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        #p[1:2] = log.(p[1:2])
        p[1:2] = logit.((p[1:2] .- lb[1:2])./(ub[1:2] .- lb[1:2]))
        p[3:4] = p[3:4]     
        
    elseif f_str == "softplus"
        
        p[1] = log(p[1])
        p[2:3] = p[2:3]
        
    end
    
    return p
    
end
