"""
    fit_jointmodel(datapath, resultspath; options)

Fit a single joint model to one trial-set or simultaneously to multiple trial-sets and save the results

Arguments:

-`datapath`: A vector of string specifying the path of the ".mat" file (s) containing the data
-`resultspath`: String specifying the path of ".mat" file where results are saved

Optional arguments:
-`options`: an instance of [`joint_options`](@ref)
"""
function fit_jointmodel(datapath::Vector{String}, resultspath::String; options::joint_options = joint_options(), verbose::Bool=true, computeHessian::Bool=true, optimizemodel::Bool=false)
    @assert SubString(resultspath, length(resultspath)-3, length(resultspath)) == ".mat"
    resultsfolderpath = splitdir(resultspath)[1]
    if !isdir(resultsfolderpath)
        mkpath(resultsfolderpath)
        @assert isdir(resultsfolderpath)
    end

    !verbose || println("Loading the data")
    options.datapath = datapath;
    data, μ_rnt = load_joint_data(options.datapath;
                            break_sim_data = options.break_sim_data,
                            centered = options.centered,
                            cut = options.cut,
                            delay = options.delay,
                            do_RBF = options.do_RBF,
                            dt = options.dt,
                            extra_pad = options.extra_pad,
                            filtSD = options.filtSD,
                            nback = options.nback,
                            nRBFs = options.nRBFs,
                            pad = options.pad,
                            pcut = options.pcut)


    optimizemodel ? nothing : options.fit_noiseless_model = false
    if verbose && options.fit_noiseless_model
        println("Computing the initial value of the parameters")
    end
    θ = θjoint(data;
               ftype = options.ftype,
               remap = options.remap,
               modeltype = options.modeltype,
               fit_noiseless_model = options.fit_noiseless_model)
   options = joint_options!(options, θ.f)
   options.x0 = flatten(θ)
   model = jointDDM(θ=θ, joint_data=data, n=options.n, cross=options.cross)

   if optimizemodel
       !verbose || println("Optimizing the model")
       model, = optimize_jointmodel(model, options)
   end

   if computeHessian
       !verbose || println("Computing the Hessian")
       H = Hessian(model)
       !verbose || println("Computing the confidence_intervals")
       CI = confidence_interval(H, model.θ)
   else
       H = Array{Float64}(undef,0,0)
       CI = Array{Float64}(undef,0,0)
   end

   !verbose || println("simulating firing rates and probability of a right choice")
   λ, fractionright, a = simulate_model(model)

   !verbose || println("Saving the results")
   save_model(resultspath, model, options;
              Hessian=H,
              CI=CI,
              μ_rnt = μ_rnt,
              λ=λ,
              fractionright=fractionright,
              a=a)

   !verbose || println("Done!")
end
