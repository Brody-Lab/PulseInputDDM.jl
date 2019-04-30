  
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