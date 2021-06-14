"""
    load_choice_data(file)

Given a path to a `.MAT` file containing data (properly formatted), loads data into
an acceptable format to use with `pulse_input_DDM` to fit its choice model.

"""
function load_choice_data(file::String; centered::Bool=false, dt::Float64=1e-2)

    data = read(matopen(file), "rawdata")

    T = vec(data["T"])
    L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
    R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))
   # gamma = vec(data["gamma"])

    choices = vec(convert(BitArray, data["pokedR"]))
    hits = vec(convert(BitArray, data["hit"]))

    if haskey(data, "sessbnd")
        sessbnd = vec(convert(BitArray, data["sessbnd"]))
    else
        sessbnd = [rand()< eps() for i in 1:length(choices)]
    end    
    sessbnd[1] = true  # marking the first trial

    theclicks = clicks.(L, R, T)	
   # theclicks = clicks.(L, R, T, gamma)
    binned_clicks = bin_clicks.(theclicks, centered=centered, dt=dt)
    inputs = map((clicks, binned_clicks, sessbnd)-> choiceinputs(clicks=clicks, binned_clicks=binned_clicks, sessbnd=sessbnd, 
        dt=dt, centered=centered), theclicks, binned_clicks, sessbnd)

    choicedata.(inputs, choices, hits)

end


"""
    save_choice_model(file, model, options)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save_choice_model(file, model, options, ll)

    @unpack lb, ub, fit = options
    @unpack θ = model

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
        "name" => get_param_names(θ), "loglik" => ll, 
        "lb"=> lb, "ub"=> ub, "fit"=> fit)

    matwrite(file, dict)

end


"""
    save_choice_model(file, model, options, CI)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save_choice_model(file, model, options, ll, CI)

    @unpack lb, ub, fit = options
    @unpack θ = model

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
        "name" => get_param_names(θ), "loglik" => ll, 
        "lb"=> lb, "ub"=> ub, "fit"=> fit,
        "CI" => CI)

    matwrite(file, dict)

    #=
    if !isempty(H)
        #dict["H"] = H
        hfile = matopen(path*"hessian_"*file, "w")
        write(hfile, "H", H)
        close(hfile)
    end
    =#

end

"""
    save_choice_model(file, model, options, CI, outsample_ll)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save_choice_model(file, model, options, ll, CI, outsample_ll, pright)

    @unpack lb, ub, fit = options
    @unpack θ = model

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
        "name" => get_param_names(θ), 
        "loglik" => ll, "outsample_ll" => outsample_ll,
        "lb"=> lb, "ub"=> ub, "fit"=> fit,
        "CI" => CI, "p_goright" => pright)

    matwrite(file, dict)

    #=
    if !isempty(H)
        #dict["H"] = H
        hfile = matopen(path*"hessian_"*file, "w")
        write(hfile, "H", H)
        close(hfile)
    end
    =#

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

    # if length(x) == 15
    #     x = vcat(x[1:5], x[5], x[6:end])
    #     lb = vcat(x[1:5], x[5], x[6:end])
    #     ub = vcat(x[1:5], x[5], x[6:end])
    #     fit = vcat(x[1:5], x[5], x[6:end])
    # end
    
    Flatten.reconstruct(θchoice(), x), choiceoptions(fit=fit, lb=lb, ub=ub)

end


"""
    bin_clicks(clicks::Vector{T})

Wrapper to broadcast bin_clicks across a vector of clicks.
"""
bin_clicks(clicks::Vector{T}; dt::Float64=1e-2, centered::Bool=false) where T <: Any =
    bin_clicks.(clicks; dt=dt, centered=centered)


"""
    bin_clicks(clicks)

Bins clicks, based on dt (defaults to 1e-2). 'centered' determines if the bin edges
occur at 0 and dt (and then ever dt after that), or at -dt/2 and dt/2 (and then
every dt after that). If the former, the bins align with the binning of spikes
in the neural model. For choice model, the latter is fine.
"""
function bin_clicks(clicks::clicks; dt::Float64=1e-2, centered::Bool=false)

    @unpack T,L,R = clicks
    nT = ceil(Int, round((T/dt), digits=10))

    if centered
        nL = searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), L)
        nR = searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), R)

    else
        nL = searchsortedlast.(Ref(0.:dt:nT*dt), L)
        nR = searchsortedlast.(Ref(0.:dt:nT*dt), R)

    end

    binned_clicks(nT, nL, nR)

end
