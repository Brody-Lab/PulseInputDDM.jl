"""
"""
function load_choice_data(path::String, file::String)

    data = read(matopen(path*file), "rawdata")

    data["T"] = data["T"]
    data["pokedR"] = vec(convert(BitArray, data["pokedR"]))
    data["correct_dir"] = vec(convert(BitArray, data["correct_dir"]))
    data["leftbups"] = map(x-> vec(collect(x)), data["leftbups"])
    data["rightbups"] = map(x-> vec(collect(x)), data["rightbups"])
    
    return data

end


"""
"""
function bin_clicks!(data::Dict; use_bin_center::Bool=false, dt::Float64=1e-2)
    
    data["dt"] = dt
    data["use_bin_center"] = use_bin_center
    
    data["nT"], data["binned_leftbups"], data["binned_rightbups"] = 
        bin_clicks(data["T"], data["leftbups"], data["rightbups"], dt=dt, use_bin_center=use_bin_center)
    
    data["ΔLRT"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"])[end], data["nT"], data["leftbups"], data["rightbups"])
    data["ΔLR"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"]), data["nT"], data["leftbups"], data["rightbups"])
    
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

    dict = Dict("ML_params"=> vcat(pz["final"], pd["final"]),
        "name" => vcat(pz["name"], pd["name"]),
        "CI_plus" => vcat(pz["CI_plus"], pd["CI_plus"]),
        "CI_minus" => vcat(pz["CI_minus"], pd["CI_minus"]),
        "lb"=> vcat(pz["lb"], pd["lb"]),
        "ub"=> vcat(pz["ub"], pd["ub"]),
        "fit"=> vcat(pz["fit"], pd["fit"]))

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

    pz["state"] = read(matopen(path*file),"ML_params")[1:dimz]
    pd["state"] = read(matopen(path*file),"ML_params")[dimz+1:dimz+2]

    return pz, pd

end
