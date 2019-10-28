"""
"""
function load_choice_data(path::String, file::String; 
                use_bin_center::Bool=false, dt::Float64=1e-2)

    println("loading data \n")
    data = read(matopen(path*file), "rawdata")

    data["T"] = data["T"]
    data["pokedR"] = vec(convert(BitArray, data["pokedR"]))
    
    mykeys = collect(keys(data))
    
    Lkey_bool = findall(map(key-> occursin("left", key), mykeys))
    Rkey_bool = findall(map(key-> occursin("right", key), mykeys))
    corkey_bool = findall(map(key-> occursin("correct", key), mykeys))
    
    data["left"] = map(x-> vec(collect(x)), data[mykeys[Lkey_bool][1]])
    data["right"] = map(x-> vec(collect(x)), data[mykeys[Rkey_bool][1]])

    if !isempty(corkey_bool)
        data["correct"] = vec(convert(BitArray, data[mykeys[corkey_bool][1]]))
    end

    data = bin_clicks!(data; use_bin_center=use_bin_center, dt=dt)
    
    return data

end


"""
"""
function bin_clicks!(data::Dict; use_bin_center::Bool=false, dt::Float64=1e-2)
    
    data["dt"] = dt
    data["use_bin_center"] = use_bin_center
    
    data["nT"], data["binned_left"], data["binned_right"] = 
        bin_clicks(data["T"], data["left"], data["right"], dt=dt, use_bin_center=use_bin_center)
    
    data["ΔLRT"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"])[end], data["nT"], data["left"], data["right"])
    data["ΔLR"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"]), data["nT"], data["left"], data["right"])
    
    return data    

end


"""
"""
function bin_clicks(T,L,R;dt::Float64=1e-2, use_bin_center::Bool=false)
    
    nT = ceil.(Int, round.((T/dt), digits=10)) 
    #added on 6/11/19, to avoid problem, such as 0.28/1e-2 = 28.0000000004, etc.

    if use_bin_center
        
        #so that a(t) is computed to middle of bin
        nL =  map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), nT, L)
        nR = map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), nT, R)
        
    else 
                   
        nL =  map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), nT, L)
        nR = map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), nT, R)
        
    end
    
    return nT, nL, nR
    
end


"""
    save_optimization_parameters(path, file, pz, pd; H=[])
Given a path and dictionaries produced by optimize_model(), save the results of the optimization to a .MAT file
"""
function save_optimization_parameters(path, file, pz, pd; H=[])

    println("done. saving ML parameters! \n")
    dict = Dict("ML_params"=> vcat(pz["final"], pd["final"]),
        "name" => vcat(pz["name"], pd["name"]),
        "lb"=> vcat(pz["lb"], pd["lb"]),
        "ub"=> vcat(pz["ub"], pd["ub"]),
        "fit"=> vcat(pz["fit"], pd["fit"]))

    if haskey(pz,"CI_plus")
        
        dict["CI_plus"] = vcat(pz["CI_plus"], pd["CI_plus"])
        dict["CI_minus"] = vcat(pz["CI_minus"], pd["CI_minus"])

    end
    
    if !isempty(H)
        dict["H"] = H
    end

    matwrite(path*file, dict)

end


"""
    reload_optimization_parameters(path, file, pz, pd)
Given a path and dictionaries, reload the results of a previous optimization saved as a .MAT file and 
place them in the "state" key of the dictionaires that optimize_model() expects.
"""
function reload_optimization_parameters(path, file, pz, pd)

    println("reloading saved ML params \n")
    pz["state"] = read(matopen(path*file),"ML_params")[1:dimz]
    pd["state"] = read(matopen(path*file),"ML_params")[dimz+1:dimz+2]

    return pz, pd

end
