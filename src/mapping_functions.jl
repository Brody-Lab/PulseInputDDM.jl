#################################### Choice observation model #################################

"""
    map_split_combine(p_opt, p_const, fit_vec, dt; map_str::String)  

    Combine constant and variable optimization components, split into functional groups andmap to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt, map_str::String)
    
    pz,bias = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec))
    #p = inv_breakup(map_pz!(pz,dt,map_str=map_str),bias) 
    pz = map_pz!(pz,dt,map_str=map_str)
    
    return pz, bias
    
end

"""
    split_combine_invmap(pz, bias, fit_vec, dt, map_str::String)  

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, bias::TT, fit_vec, dt, map_str::String) where {TT <: Any}

    pz = inv_map_pz!(copy(pz),dt,map_str=map_str)     
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,bias),fit_vec)

    return p_opt, p_const
    
end

split_latent_and_observation(p::Vector{TT}) where {TT} = p[1:dimz],p[dimz+1]
combine_latent_and_observation(pz::Vector{TT},bias::TT) where {TT} = vcat(pz,bias)
    
    
#################################### Poisson neural observation model #########################

"""
    map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, map_str::String)  

    Combine constant and variable optimization components, split into functional groups and map to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, map_str::String, N::Int, dimy::Int)

    pz,py = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec), N, dimy)
    pz = map_pz!(pz,dt,map_str=map_str)       
    py = map_py!.(py,f_str=f_str)
    
    return pz,py
    
end

"""
    split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, fit_vec, dt, f_str::String, map_str::String)

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, fit_vec, dt, f_str::String, 
        map_str::String) where {TT <: Any}

    pz = inv_map_pz!(copy(pz), dt, map_str=map_str)     
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
    
function map_py!(p::Vector{TT};f_str::String="softplus",map_str::String="exp") where {TT}
        
    if f_str == "exp"
        
        p[1] = exp(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = exp.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = 1e-5 + 99.99 * 0.5*(1+tanh.(p[1:2]))
            p[3:4] = -9.99 + 9.99*2 * 0.5*(1+tanh.(p[3:4]))
        end
        
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

function inv_map_py!(p::Vector{TT};f_str::String="softplus",map_str::String="exp") where {TT}
     
    if f_str == "exp"

        p[1] = log(p[1])
        p[2] = p[2]
        
    elseif f_str == "sig"
        
        if map_str == "exp"
            p[1:2] = log.(p[1:2])
            p[3:4] = p[3:4]
        elseif map_str == "tanh"
            #fix this
            p[1:2] = atanh.(((p[1:2] - 1e-5)/(99.99*0.5)) - 1)
            p[3:4] = atanh.(((p[3:4] + 9.99)/(9.99*2*0.5)) - 1)
        end
        
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

#################################### Poisson neural observation model w/ RBF #########################

function split_latent_and_observation(p::Vector{T}, N::Int, dimy::Int, numRBF::Int) where {T <: Any}
                
    pz = p[1:dimz]
    py = reshape(p[dimz+1:dimz+dimy*N],dimy,N)
    py = map(i->py[:,i],1:N)
    pRBF = reshape(p[dimz+dimy*N+1:dimz+dimy*N+numRBF*N],numRBF,N)
    pRBF = map(i->pRBF[:,i],1:N)

    return pz, py, pRBF
    
end

combine_latent_and_observation(pz,py,pRBF) = vcat(pz,vcat(py...),vcat(pRBF...))

"""
    split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, pRBF::Vector{Vector{TT}}, fit_vec, dt, f_str::String, map_str::String)

    Inverse map parameters to unbounded domain for optimization, combine functional groups and split into optimization variables and constants
"""
function split_combine_invmap(pz::Vector{TT}, py::Vector{Vector{TT}}, pRBF::Vector{Vector{TT}},
        fit_vec, dt, f_str::String, 
        map_str::String) where {TT <: Any}

    pz = inv_map_pz!(copy(pz), dt, map_str=map_str)     
    py = inv_map_py!.(deepcopy(py), f_str=f_str)
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz,py,pRBF),fit_vec)
    
    return p_opt, p_const
    
end

"""
    map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, map_str::String)  

    Combine constant and variable optimization components, split into functional groups and map to bounded domain
"""
function map_split_combine(p_opt, p_const, fit_vec, dt, f_str::String, map_str::String, N::Int, dimy::Int, numRBF::Int)

    pz,py,pRBF = split_latent_and_observation(combine_variable_and_const(p_opt, p_const, fit_vec), N, dimy, numRBF)
    pz = map_pz!(pz,dt,map_str=map_str)       
    py = map_py!.(py,f_str=f_str)
    
    return pz,py,pRBF
    
end
      
#################### Common functions used in wrappers for all models ###########################
     
function map_pz!(x,dt;map_str::String="exp")
    
    lb = [eps(), 10., -5., eps(), eps(), eps(), eps()]
    ub = [10., 100, 5., 800., 40., 2., 10.]
    
    x[3] = lb[3] + (ub[3] - lb[3]) .* normtanh.(x[3])
    
    if map_str == "exp"
        x[[1,2,4,5,6,7]] = lb[[1,2,4,5,6,7]] + exp.(x[[1,2,4,5,6,7]])
    elseif map_str == "tanh"        
        x[[1,2,4,5,6,7]] = lb[[1,2,4,5,6,7]] + (ub[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]]) .* normtanh.(x[[1,2,4,5,6,7]])        
    end
        
    return x
    
end

function inv_map_pz!(x,dt;map_str::String="exp")
    
    lb = [eps(), 10., -5., eps(), eps(), eps(), eps()]
    ub = [10., 100, 5., 800., 40., 2., 10.]
    
    x[3] = normatanh.((x[3] - lb[3])./(ub[3] - lb[3]))
    
    if map_str == "exp"
        x[[1,2,4,5,6,7]] = log.(x[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]])
    elseif map_str == "tanh"
        x[[1,2,4,5,6,7]] = normatanh.((x[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]])./(ub[[1,2,4,5,6,7]] - lb[[1,2,4,5,6,7]]))
    
    end
        
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