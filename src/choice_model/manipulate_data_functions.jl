##### Choice observation model #################################

function load_choice_data(path::String, file::String)

    data = read(matopen(path*file), "rawdata")

    data["T"] = data["T"]
    data["pokedR"] = vec(convert(BitArray, data["pokedR"]))
    data["correct_dir"] = vec(convert(BitArray, data["correct_dir"]))
    data["leftbups"] = map(x-> vec(collect(x)), data["leftbups"])
    data["rightbups"] = map(x-> vec(collect(x)), data["rightbups"])
    
    return data

end

function aggregate_choice_data(path::String, sessids::Vector{Vector{Int}}, ratnames::Vector{String})
    
    data = Dict{String,Any}("leftbups" => Vector{Vector{Float64}}(), "rightbups" => Vector{Vector{Float64}}(), 
            "T" => Vector{Float64}(), 
            "pokedR" => Vector{Bool}(), "correct_dir" => Vector{Bool}(), 
            "sessID" => Vector{Int}(), "ratID" => Vector{String}(),
            "stim_start" => Vector{Float64}(), "cpoke_end" => Vector{Float64}())
    
    for j = 1:length(ratnames)
        for i = 1:length(sessids[j])
            rawdata = read(matopen(path*"/"*ratnames[j]*"_"*string(sessids[j][i])*".mat"),"rawdata")
            data = append_choice_data!(data,rawdata)
        end
    end
    
    return data
    
end

#=function aggregate_choice_data(path::String, files)
    
    data = Dict{String,Any}("leftbups" => Vector{Vector{Float64}}(), 
            "rightbups" => Vector{Vector{Float64}}(), 
            "T" => Vector{Float64}(), 
            "pokedR" => Vector{Bool}(), 
            "correct_dir" => Vector{Bool}())
    
    for i = 1:length(files)
        rawdata = read(matopen(path*files[i]),"rawdata")
        data = append_choice_data!(data,rawdata)
    end
    
    return data
    
end=#

function append_choice_data!(data::Dict, rawdata::Dict)

    ntrials = length(rawdata["T"])

    append!(data["T"], rawdata["T"])
    append!(data["pokedR"], vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"], vec(convert(BitArray,rawdata["correct_dir"])))
    
    append!(data["leftbups"], map(x-> vec(collect(x)), rawdata["leftbups"]))
    append!(data["rightbups"], map(x-> vec(collect(x)), rawdata["rightbups"]))
    
    return data

end

function bin_clicks!(data::Dict, use_bin_center::Bool; dt::Float64=1e-2)
    
    data["dt"] = dt
    data["use_bin_center"] = use_bin_center
    
    data["nT"], data["binned_leftbups"], data["binned_rightbups"] = 
        bin_clicks(data["T"], data["leftbups"], data["rightbups"], dt, use_bin_center)
    
    data["ΔLRT"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"])[end],data["nT"],data["leftbups"],data["rightbups"])
    data["ΔLR"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"]),data["nT"],data["leftbups"],data["rightbups"])
    
    return data    

end

function bin_clicks(T,L,R,dt,use_bin_center)
    
    #binnedT = ceil.(Int,T/dt)
    binnedT = ceil.(Int,round.((T/dt) ./1e-10) .*1e-10) 
    #added on 6/11/19, to avoid problem, such as 0.28/1e-2 = 28.0000000004, etc.

    nT = binnedT
    #nL =  map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, L)
    #nR = map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, R)
    
    if use_bin_center
        
        #changed on 6/9, based on approaches developed with JP, 
        #so that a(t) is computed to middle of bin, and spikes are computed within bin edges
        #nL =  map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), binnedT, L)
        #nR = map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), binnedT, R)
        nL =  map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), binnedT, L)
        nR = map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), binnedT, R)
        
    else 
                   
        #this is for data generation
        nL =  map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), binnedT, L)
        nR = map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), binnedT, R)
        
    end
    
    return nT, nL, nR
    
end
