"""
    save_choice_data(file)

Given a path, save data in to a `.MAT` file of an acceptable format, containing data,
to use with `PulseInputDDM` to fit its choice model.
"""
function save_choice_data(file::String, data)
    
    rawdata = Dict("rawdata" => [(leftbups = x.click_data.clicks.L, rightbups = x.click_data.clicks.R, 
        T = x.click_data.clicks.T, pokedR = x.choice) for x in data])   
    matwrite(file, rawdata)

end


"""
    load_choice_data(file)

Given a path to a `.MAT` file containing data (properly formatted), loads data into
an acceptable format to use with `pulse_input_DDM` to fit its choice model.

"""
function load_choice_data(file::String; centered::Bool=false, dt::Float64=1e-2)
    
    data = read(matopen(file), "rawdata")

    if typeof(data) .== Dict{String, Any}
    
        T = vec(data["T"])
        L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
        R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))
        choices = vec(convert(BitArray, data["pokedR"]))
        
    elseif typeof(data) == Vector{Any}
        
        T = vec([data[i]["T"] for i in 1:length(data)])
        L = vec(map(x-> vec(collect((x[collect(keys(x))[occursin.("left", collect(keys(x)))][1]]))), data))
        R = vec(map(x-> vec(collect((x[collect(keys(x))[occursin.("right", collect(keys(x)))][1]]))), data))
        choices = vec(convert(BitArray, [data[i]["pokedR"] for i in 1:length(data)]))
            
    end
    
    theclicks = clicks.(L, R, T)
    binned_clicks = bin_clicks.(theclicks, centered=centered, dt=dt)
    inputs = map((clicks, binned_clicks)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, 
        dt=dt, centered=centered), theclicks, binned_clicks)

    choicedata.(inputs, choices)

end


"""
    save_choice_model(file, model, options)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save_choice_model(file, model)

    @unpack lb, ub, fit, θ = model

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
        "name" => ["σ2_i", "B", "λ", "σ2_a", "σ2_s", "ϕ", "τ_ϕ", "bias", "lapse"],
        "lb"=> lb, "ub"=> ub, "fit"=> fit)

    matwrite(file, dict)

end


"""
    reload_choice_model(file)
    
Given a path, reload the results of a previous optimization saved as a .MAT file and
place them in the "state" key of the dictionaires that optimize_model() expects.
"""
function reload_choice_model(file)

    x = read(matopen(file), "ML_params")
    lb = read(matopen(file), "lb")
    ub = read(matopen(file), "ub")
    fit = read(matopen(file), "fit")
    
    choiceDDM(θ=Flatten.reconstruct(θchoice(), x), fit=fit, lb=lb, ub=ub)
    
end
