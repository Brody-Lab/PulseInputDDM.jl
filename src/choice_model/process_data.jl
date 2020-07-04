"""
    load(file)

Given a path to a .mat file containing data (properly formatted), loads data into
an acceptable format to use with pulse_input_DDM.
"""
function load(file::String; centered::Bool=false, dt::Float64=1e-2)

    data = read(matopen(file), "rawdata")

    T = vec(data["T"])
    L = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]]))
    R = vec(map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]]))
    choices = vec(convert(BitArray, data["pokedR"]))
    sessbnd = vec(convert(BitArray, data["sessbnd"]))

    click_times = clicks.(L, R, T)
    binned_clicks = bin_clicks.(click_times, centered=centered, dt=dt)
    inputs = choiceinputs.(click_times, binned_clicks, dt, centered)

    choicedata.(inputs, choices, sessbnd)

end


"""
    save(file, model, options, CI)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save(file, model, options, modeltype, ll, prob_right; CI = 0)

    @unpack lb, ub, fit = options
    @unpack θ = model

    if modeltype == "expfilter"
        name = ["B", "λ", "σ2_i", "σ2_a", "σ2_s", "ϕ", "τ_ϕ", "h_eta","h_beta","h_drift_scale","bias", "lapse"]
    elseif modeltype == "expfilter_ce"
        name = ["B", "λ", "σ2_i", "σ2_a", "σ2_s", "ϕ", "τ_ϕ", "h_etaC","h_etaE", "h_betaC", "h_betaE","h_drift_scale", "bias", "lapse"]
    else
        error("Unknown model identifier $modeltype")
    end

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
                "name" => name, "loglikelihood" => ll,
                "lb"=> lb,
                "ub"=> ub,
                "fit"=> fit,
                "modeltype"=> modeltype,
                "prob_right"=> prob_right,
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
    reload(file)

Given a path and dictionaries, reload the results of a previous optimization saved as a .MAT file and
place them in the "state" key of the dictionaires that optimize_model() expects.
"""
function reload(file)

    read(matopen(file), "ML_params")

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
