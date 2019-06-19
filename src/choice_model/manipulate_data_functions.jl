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
    
    return data

end

function bin_clicks!(data::Dict; dt::Float64=1e-2)
    
    data["dt"] = dt
    
    data["nT"], data["binned_leftbups"], data["binned_rightbups"] = 
        bin_clicks(data["T"], data["leftbups"], data["rightbups"], dt)
    
    return data    

end

function bin_clicks(T,L,R,dt)
    
    #binnedT = ceil.(Int,T/dt)
    binnedT = ceil.(Int,round.((T/dt) ./1e-10) .*1e-10) 
    #added on 6/11/19, to avoid problem, such as 0.28/1e-2 = 28.0000000004, etc.

    nT = binnedT
    nL =  map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, L)
    nR = map((x,y)-> vec(qfind(0.:dt:x*dt,y)), binnedT, R)
    
    return nT, nL, nR
    
end
