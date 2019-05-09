##### Choice observation model #################################

function aggregate_choice_data(path::String, sessids::Vector{Vector{Int}}, ratnames::Vector{String})
    
    data = Dict{String,Any}("leftbups" => Vector{Vector{Float64}}(), "rightbups" => Vector{Vector{Float64}}(), 
            "T" => Vector{Float64}(), 
            "pokedR" => Vector{Bool}(), "correct_dir" => Vector{Bool}(), 
            "sessID" => Vector{Int}(), "ratID" => Vector{String}(),
            "stim_start" => Vector{Float64}(), "cpoke_end" => Vector{Float64}())
    
    for j = 1:length(ratnames)
        for i = 1:length(sessids[j])
            rawdata = read(matopen(path*"/"*ratnames[j]*"_"*string(sessids[j][i])*".mat"),"rawdata")
            data = append_choice_data!(data,rawdata,ratnames[j],sessids[j][i])
        end
    end
    
    return data
    
end

function append_choice_data!(data::Dict, rawdata::Dict, ratname::String, sessID::Int)

    ntrials = length(rawdata["T"])

    append!(data["T"], rawdata["T"])
    append!(data["pokedR"], vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"], vec(convert(BitArray,rawdata["correct_dir"])))
    
    append!(data["leftbups"], map(x-> vec(collect(x)), rawdata["leftbups"]))
    append!(data["rightbups"], map(x-> vec(collect(x)), rawdata["rightbups"]))
    append!(data["sessID"], repeat([sessID], inner=ntrials))
    append!(data["ratID"], repeat([ratname], inner=ntrials))
    
    #removed because I didn't tell Chuck to put it in, and not currently necessary in the model fitting
    #append!(data["stim_start"], rawdata["stim_start"])
    #append!(data["cpoke_end"], rawdata["cpoke_end"])

    return data

end

function bin_clicks!(data::Dict; dt::Float64=1e-2)
    
    data["dt"] = dt
    
    binnedT = ceil.(Int,data["T"]/dt);

    data["nT"] = binnedT
    data["binned_leftbups"] =  map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, data["leftbups"])
    data["binned_rightbups"] = map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, data["rightbups"])
    
    return data    

end
