module opt_diff_model

using module_DDM_v4, MAT

export make_data, my_sigmoid_2, LL_func, ll_func

function make_data(pth,sessid,ratname;dt=1e-3)

    data = Dict("leftbups" => Vector{Vector{Float64}}(), "rightbups" => Vector{Vector{Float64}}(), 
                "binned_leftbups" => Vector{Vector{Int64}}(), "binned_rightbups" => Vector{Vector{Int64}}(),
                "T" => Vector{Float64}(), "nT" => Vector{Int64}(), 
                "pokedR" => Vector{Bool}(), "correct_dir" => Vector{Bool}(), 
                "spike_counts" => Vector{Vector{Vector{Int}}}(),"N" => Vector{UnitRange{Int64}}(),
                "trial" => Vector{UnitRange{Int64}}(),
                "sessid" => Vector{Int}(), "N0" => 0, "trial0" => 0);
    
    in_pth = pth*"/data/hanks_data_sessions"

    for i = 1:length(sessid)
        file = matopen(in_pth*"/"*ratname*"_"*string(sessid[i])*".mat")
        rawdata = read(file,"rawdata")
        data = package_data!(data,rawdata,"spikes",dt=dt,organize="by_neuron")
    end

    N = data["N0"];
    
    return data, N
    
end

function ll_func{TT}(p::Vector{TT},k::Vector{Vector{Int}},trials::UnitRange{Int64},ΔLR::Vector{Vector{Int}};dt::Float64=1e-3)
        
        k = vcat(k...)
        ΔLR = vcat(ΔLR[trials]...)
        LL_func(p,k,ΔLR,dt=dt)
    
end

function LL_func{TT}(p::Vector{TT}, k::Vector{Int}, ΔLR::Vector{Int}; kind::String = "exp", dt::Float64=1e-3)
        
    λ = my_sigmoid_2(ΔLR,map_func_fr!(p,kind));      
    LL = reduce(+, poiss_likelihood(k,λ,dt))  
    
    return LL
    
end

function my_sigmoid_2(x,p)
    
    temp = -p[3]*x + p[4]

    y = p[1] + p[2]./(1. + exp.(temp));
     
    #protect from NaN gradient values
    y[exp.(temp) .<= 1e-150] = p[1] + p[2]
    y[exp.(temp) .>= 1e150] = p[1]
    
    return y
    
end

function poiss_likelihood(k,λ,dt)
    
    LL = k.*log.(λ*dt) - λ*dt - lgamma.(k+1)
    
    return LL
    
end

end