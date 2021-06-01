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

    loadedoptions = read(matopen(datapath), "options")
    options = DDLMoptions(  a_bases = map(x->vec(x), vec(loadedoptions["a_bases"])),
                            centered = loadedoptions["centered"],
                            cross = loadedoptions["cross"],
                            datapath = datapath,
                            dt = loadedoptions["dt"],
                            fit = convert(BitArray, vec(loadedoptions["fit"])),
                            L2regularizer = loadedoptions["L2regularizer"],
                            lb = vec(loadedoptions["lb"]),
                            n = convert(Int64, loadedoptions["n"]),
                            nback = convert(Int64, loadedoptions["nback"]),
                            npostpad_abar = convert(Int64, loadedoptions["npostpad_abar"]),
                            remap = loadedoptions["remap"],
                            resultspath = loadedoptions["resultspath"],
                            ub = vec(loadedoptions["ub"]),
                            x0 = vec(loadedoptions["x0"]))
    isfile(options.resultspath) ? x = vec(read(matopen(options.resultspath),"ML_params")) : x = options.x0
    trialsets = map(x->parse_one_trialset(x, options), vec(read(matopen(datapath), "trialsets")))
    DDLM(data=trialsets, options=options, θ=θDDLM(x))
end

"""
    parse_one_trialset

Parse the trial set data exported from MATLAB

INPUT

-trialset: data corresponding to a single trial set

OUTPUT

-an instance of trialsetdata
"""
function parse_one_trialset(trialset::Dict, options::DDLMoptions)
    rawtrials = vec(trialset["trials"])
    rawclicktimes = map(x->x["clicktimes"], rawtrials)

    L = map(x->vec(x["L"]), rawclicktimes)
    isscalar = map(x->typeof(x),L).==Float64
    L[isscalar] = map(x->[x], L[isscalar])
    L = convert(Array{Array{Float64,1},1},L)

    R = map(x->vec(x["R"]), rawclicktimes)
    isscalar = map(x->typeof(x),R).==Float64
    R[isscalar] = map(x->[x], R[isscalar])
    R = convert(Array{Array{Float64,1},1},R)

    T = map(x->x["T"], rawclicktimes)
    choice = map(x->x["choice"], rawtrials)

    clicktimes = map((L,R,T)->clicks(L=L, R=R, T=T), L, R, T)
    clickcounts = map(x->bin_clicks(x, centered=options.centered, dt=options.dt), clicktimes)
    trials = map((clickcounts, clicktimes, choice)->trialdata(clickcounts=clickcounts, clicktimes=clicktimes, choice=choice), clickcounts, clicktimes, choice)

    rawunits = vec(trialset["units"])
    Xautoreg = map(x->x["Xautoreg"], rawunits)
    y = map(x->vec(x["y"]), rawunits)
    units = map((X,y)->unitdata(X,y), Xautoreg, y)

    shifted = trialshifted(convert(Matrix{Int64}, trialset["shifted"]["choice"]),
                           convert(Matrix{Int64}, trialset["shifted"]["reward"]),
                           convert(Matrix{Int64}, trialset["shifted"]["shift"]))

    trialsetdata(shifted=shifted, trials=trials, units=units, Xtiming=trialset["Xtiming"])
end

"""
    save

Save the maximum likelihood parameters, Hessian, and in-sample predictions of a drift-diffusion linear model

ARGUMENT

-model: an instance of `DDLM`, a drift-diffusion linear model

"""

function save(model::DDLM)

    abar, choicelikelihood, Xa = pulse_input_DDM.predict_in_sample(model)
    dict = Dict("ML_params"=> pulse_input_DDM.flatten(model.θ),
                "parameter_name" => pulse_input_DDM.θDDLM_names(), #"Hessian" => Hessian(model),
                "abar_insample" => abar,
                "choicelikelihood_insample" => choicelikelihood,
                "Xa_insample" => Xa)
    matwrite(model.options.resultspath, dict)
end

"""
    θDDLM_names

Return a vector of strings for MATLAB that indicate the names of the DDLM parameters

"""
function θDDLM_names()
    ["alpha";
     "B";
     "bias";
     "k";
     "lambda";
     "lapse";
     "phi";
     "sigma2_a";
     "sigma2_i";
     "sigma2_s";
     "tau_phi"]
end
