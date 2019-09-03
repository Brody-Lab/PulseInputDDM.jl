function append_choice_data_marino!(data::Dict, rawdata::Dict)

    ntrials = length(rawdata["T"])

    append!(data["T"], rawdata["T"])
    append!(data["pokedR"], vec(convert(BitArray,rawdata["pokedR"])))
    append!(data["correct_dir"], vec(convert(BitArray,rawdata["correct_dir"])))
    append!(data["context_loc"], vec(convert(BitArray,rawdata["context_loc"])))
    
    append!(data["leftbups_loc"], map(x-> vec(collect(x)), rawdata["leftbups_loc"]))
    append!(data["rightbups_loc"], map(x-> vec(collect(x)), rawdata["rightbups_loc"]))
    append!(data["leftbups_freq"], map(x-> vec(collect(x)), rawdata["leftbups_freq"]))
    append!(data["rightbups_freq"], map(x-> vec(collect(x)), rawdata["rightbups_freq"]))
    
    return data

end