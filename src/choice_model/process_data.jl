"""
"""
function load(file::String; centered::Bool=false, dt::Float64=1e-2)

    data = read(matopen(file), "rawdata")

    T = vec(data["T"])
    L = map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("left", collect(keys(data)))][1]])
    R = map(x-> vec(collect(x)), data[collect(keys(data))[occursin.("right", collect(keys(data)))][1]])
    choices = vec(convert(BitArray, data["pokedR"]))

    click_times = clicks.(L, R, T)
    binned_clicks = bin_clicks.(click_times, centered=centered, dt=dt)
    inputs = choiceinputs.(click_times, binned_clicks, dt, centered)

    choicedata.(inputs, choices)

end


"""
"""
bin_clicks(clicks::Vector{T}; dt::Float64=1e-2, centered::Bool=false) where T <: Any =
    bin_clicks.(clicks; dt=dt, centered=centered)


"""
"""
function bin_clicks(clicks::clicks; dt::Float64=1e-2, centered::Bool=false)

    @unpack T,L,R = clicks
    nT = ceil(Int, round((T/dt), digits=10))
    #added on 6/11/19, to avoid problem, such as 0.28/1e-2 = 28.0000000004, etc.

    if centered

        #so that a(t) is computed to middle of bin
        #nL =  map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), nT, L)
        #nR = map((x,y)-> map(z-> searchsortedlast((0. -dt/2):dt:(x -dt/2)*dt,z), y), nT, R)
        #nL =  map(z-> searchsortedlast((0. -dt/2):dt:(nT -dt/2)*dt,z), L)
        #nR = map(z-> searchsortedlast((0. -dt/2):dt:(nT -dt/2)*dt,z), R)
        nL =  searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), L)
        nR = searchsortedlast.(Ref((0. -dt/2):dt:(nT -dt/2)*dt), R)

    else

        #nL =  map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), nT, L)
        #nR = map((x,y)-> map(z-> searchsortedlast(0.:dt:x*dt,z), y), nT, R)
        #nL =  map(z-> searchsortedlast(0.:dt:nT*dt,z), L)
        #nR = map(z-> searchsortedlast(0.:dt:nT*dt,z), R)
        nL =  searchsortedlast.(Ref(0.:dt:nT*dt), L)
        nR = searchsortedlast.(Ref(0.:dt:nT*dt), R)

    end

    #binned_clicks(clicks=clicks, nT=nT, nL=nL, nR=nR, dt=dt, centered=centered)
    binned_clicks(nT, nL, nR)

    #data["ΔLRT"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"])[end], data["nT"], data["leftbups"], data["rightbups"])
    #data["ΔLR"] = map((nT,L,R)-> diffLR(nT,L,R,data["dt"]), data["nT"], data["leftbups"], data["rightbups"])

end


"""
    save_optimization_parameters(file, model, options, CI)

Given a file, model produced by optimize and options, save the results of the optimization to a .MAT file
"""
function save(file, model, options, CI)

    @unpack lb, ub, fit = options
    @unpack θ = model

    dict = Dict("ML_params"=> collect(Flatten.flatten(θ)),
        "name" => ["σ2_i", "B", "λ", "σ2_a", "σ2_s", "ϕ", "τ_ϕ", "bias", "lapse"],
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
    reload_optimization_parameters(file)
Given a path and dictionaries, reload the results of a previous optimization saved as a .MAT file and
place them in the "state" key of the dictionaires that optimize_model() expects.
"""
function reload(file)

    read(matopen(file), "ML_params")

end
