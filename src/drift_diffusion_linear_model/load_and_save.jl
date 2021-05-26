"""
    load_DDLM(datapath::String)

Initiate a drift-diffusion linear model by loading the data and specifications from a MATLAB file

The file `<datapath>.mat` should contain two structures, `options` and `trialsets.` The structure `options` should contains the same fields as in the type `DDLMoptions` and the structure `trialsets` should be a vector whose each element is an instance of `trialsetdata.` Each element of `trialsets` correspond to one trial-set and has the fields `trials`, `units`, and `trialshifted.` The fields `trials,` `units,` and `shifted` are each a vector of instances of `trialdata,` `unitdata,` and `shifted,` respectively and should be organized similarly to the types defined in the Julia `pulse_input_DDM` package.

ARGUMENT

- `datapath`: path to the file you want to load.

RETURN

- an instance of `DDLM`
"""
function load_DDLM(datapath::String)

    loadedoptions = read(matopen(filepath), "options")
    options = DDLMoptions(  a_bases = loadedoptions["a_bases"],
                            centered = loadedoptions["centered"],
                            cross = loadedoptions["cross"],
                            datapath = datapath,
                            dt = loadedoptions["dt"],
                            fit = vec(loadedoptions["fit"]),
                            L2regularizer = loadedoptions["L2regularizer"],
                            lb = vec(loadedoptions["lb"]),
                            n = loadedoptions["n"],
                            nback = loadedoptions["nback"],
                            remap = loadedoptions["remap"],
                            resultspath = loadedoptions["resultspath"],
                            ub = vec(loadedoptions["ub"]),
                            x0 = vec(loadedoptions["x0"]))
    isempty(resultspath) ? θ=DDLM(options.x0) : θ=read(matopen(resultspath),"ML_params")

    trialsets = read(matopen(filepath), "trialsets")

    L = map(x->map(y->y["L"], x["clicks"]), trialsets["trials"])
    R = map(x->map(y->y["R"], x["clicks"]), trialsets["trials"])
    T = map(x->map(y->y["T"], x["clicks"]), trialsets["trials"])
    choice = map(x->x["choice"], trialsets["trials"])

    L = vec(map(x->vec(map(x->vec(x), x)), L))
    R = vec(map(x->vec(map(x->vec(x), x)), R))
    T = vec(map(x->vec(x), T))
    choice = vec(map(x->vec(x), choice))

    clicktimes = map((L,R,T)->map((L,R,T)->clicks(L=L, R=R, T=T), L, R, T), L, R, T)
    clickcounts = map(x->map(x->bin_clicks(x, centered=options.centered, dt=options.dt), x), click_times)

    trials = map((x,y,z)->map((x,y,z)->trialdata(clickcounts=x, clicktimes=y, choice=z), x,y,z), clickcounts, clicktimes, choice)

    units = map(units->vec(map((X,y)->unitdata(X,vec(y)), units["Xautoreg"], units["y"])), vec(trialsets["units"]))

    shifted = map(x->trialshifted(x["choice"], x["reward"], vec(x["shift"])), vec(trialsets["shifted"]))

    data = map((w,x,y,z)->trialsetdata(w,x,y,z), shifted, trials, units, vec(trialsets["Xtiming"]))

    DDLM(data=data, options=options, θ=θ)
end

"""
    save

Save the maximum likelihood parameters, Hessian, and in-sample predictions of a drift-diffusion linear model

ARGUMENT

-model: an instance of `DDLM`, a drift-diffusion linear model

"""

function save(model::DDLM)

    abar_insample, choiceprobability_insample, Xa_insample = predict_in_sample(model)
    dict = Dict("ML_params"=> vec(model.θ),
                "parameter_name" => θDDLM_names(),
                "Hessian" => Hessian(model),
                "abar_insample" => abar_insample,
                "choiceprobability_insample" => choice_insample,
                "Xa_insample" => Xa_insample)
    matwrite(DDLM.options.resultspath, dict)
end

"""
    θDDLM_names

Return a vector of strings for MATLAB that indicate the names of the DDLM parameters

"""
function θDDLM_names()
    ["sigma2_i";
     "B";
     "lambda";
     "sigma2_a";
     "sigma2_s";
     "phi";
     "tau_phi";
     "alpha";
     "k";
     "bias";
     "lapse"]
end

"""
    fit_DDLM

Load the data and specifications, optimize, and save the results

"""

function fit_DDLM(datapath::String)
    model = load_DDLM(datapath)
    model = optimize(model)
    save(model)
end
