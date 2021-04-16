"""
    save_model(file, model, options)

Save results to a `.MAT` file such that `reload_jointDDM` can recreate the model in a Julia workspace or they can be loaded in MATLAB.

See also: [`reload_jointDDM`](@ref)

Arguments:
-`resultspath`: the path where the results are saved
-`model`: an instance of `jointDDM`
-`options`: an instance of `joint_options`

Optional arguments:
-`Hessian`
-`CI`: confidence intervals, a matrix whose first and second columns represent the lower and upper bounds, respectively

"""
function save_model(resultspath::String, model::jointDDM, options::joint_options; Hessian::Matrix{T} = Array{Float64}(undef,0,0), CI::Matrix{T}=Array{Float64}(undef,0,0), λ::AbstractArray=Vector{Vector{Vector{Float64}}}[], fractionright::AbstractArray=Vector{Vector{Float64}}[]) where {T <: Real}

    @unpack θ, joint_data, n, cross = model
    @unpack f = θ

    dict = Dict("ML_params"=> collect(pulse_input_DDM.flatten(θ)),
                "parameter_name" => vcat(String.(get_jointDDM_θlatent_names()), vcat(vcat(f...)...)),
                "options" => convert_to_matwritable_Dict(options),
                "CI" => CI,
                "Hessian" => Hessian,
                "expected_firing_rates" => λ,
                "fractionright" => fractionright)
    matwrite(resultspath, dict)
end
"""
    convert_to_matwritable_Dict(options)

Convert an instance of `joint_options` to a Dictionary that can be saved to a MATLAB ".mat" file
"""

function convert_to_matwritable_Dict(options::joint_options)
    dict = Dict()
    for field in fieldnames(joint_options)
        value = getfield(options, field)
        field == :modeltype ? value = String(value) : nothing
        dict[String(field)] = value
    end
    return dict
end

"""
    load_joint_options

Load an instance of `joint_options` from file

Arguments:
-`filepath`: A string

Optional arguments:
-`variablename`: name of the variable corresponding to the instance of `joint_options`
"""
function load_joint_options(filepath::String; variablename::String = "options")
    dict = read(matopen(filepath), variablename)
    options = joint_options()
    for key in keys(dict)
        value = dict[key]
        setfield!(options, Symbol(key), value)
    end
    return options
end

"""
    load_joint_model(resultspath)

Reconstruct an instance of `jointDDM` from a MAT file saved using [`save_model`](@ref)

"""
function load_joint_model(resultspath::String)
    matfile = matopen(file);
    options = load_joint_options(resultspath)
    joint_data, = load_joint_data(options.datapath;
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
    f = specify_a_to_firing_rate_function_type(joint_data; ftype=options.ftype)
    x = read(matopen(file), "ML_params")
    θ = θjoint(x,f)
    jointDDM(joint_data=joint_data, θ=θ, n=options.n, cross=options,cross)
end
