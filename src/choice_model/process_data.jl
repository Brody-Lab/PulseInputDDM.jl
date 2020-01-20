"""
"""
function load_choice_data(path::String, file::String;
                use_bin_center::Bool=false, dt::Float64=1e-4)

    println("loading data \n")
    data = read(matopen(path*file), "rawdata")

    data = process_click_input_data!(data)
    data = process_choice_data!(data)
    data = bin_clicks!(data; use_bin_center=use_bin_center, dt=dt)

    return data

end


"""
"""
function process_choice_data!(data)

    data["pokedR"] = vec(convert(BitArray, data["pokedR"]))

#    if !isempty(occursin.("correct", collect(keys(data))))
#        data["correct"] = vec(convert(BitArray, data[collect(keys(data))[occursin.("correct", collect(keys(data)))][1]]))
#    end

    return data

end


"""
"""
function process_click_input_data!(data)

    data["T"] = vec(data["T"])
    data["leftbups"] = map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]])
    data["rightbups"] = map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]])

    return data

end


"""
"""
function bin_clicks!(data::Dict; use_bin_center::Bool=false, dt::Float64=1e-4)

    data["dt"] = dt
    data["use_bin_center"] = use_bin_center

    data["leftbups"] = map((x,y) -> x[x.<y], data["leftbups"], data["T"])
    data["rightbups"] = map((x,y) -> x[x.<y], data["rightbups"], data["T"])

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

    println("done. saving ML parameters! \n")
    dict = Dict("ML_params"=> vcat(pz["final"], pd["final"]),
        "name" => vcat(pz["name"], pd["name"]),
        "lb"=> vcat(pz["lb"], pd["lb"]),
        "ub"=> vcat(pz["ub"], pd["ub"]),
        "fit"=> vcat(pz["fit"], pd["fit"]))

    if haskey(pz,"CI_plus_LRtest")

        dict["CI_plus_LRtest"] = vcat(pz["CI_plus_LRtest"], pd["CI_plus_LRtest"])
        dict["CI_minus_LRtest"] = vcat(pz["CI_minus_LRtest"], pd["CI_minus_LRtest"])

    end

    if haskey(pz,"CI_plus_hessian")

        dict["CI_plus_hessian"] = vcat(pz["CI_plus_hessian"], pd["CI_plus_hessian"])
        dict["CI_minus_hessian"] = vcat(pz["CI_minus_hessian"], pd["CI_minus_hessian"])

    end

    if !isempty(H)
        #dict["H"] = H
        hfile = matopen(path*"hessian_"*file, "w")
        write(hfile, "H", H)
        close(hfile)
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
