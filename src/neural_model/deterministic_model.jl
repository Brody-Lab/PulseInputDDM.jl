"""
"""
function optimize_model_deterministic(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e3))

    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])

    #check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,py,data,f_str; state="state")

    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace, outer_iterations=outer_iterations)
    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]

    return pz, py, converged

    return pz, py

end


"""
"""
function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)
    -compute_LL(pz, py, data, f_str)

end


"""
"""
function split_opt_params_and_close(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], py[state]), fit_vec)
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec),
        py["cells_per_session"], py["dimy"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str)

    return p_opt, ll, parameter_map_f

end

function loglikelihood(model::neuralDDM)

    @unpack θ, data = model
    @unpack θz, θy = θ

    sum(map((θy, data) -> sum(pmap(data-> loglikelihood(θz, θy, data), data,
        batch_size=length(data))), θy, data))

end


"""
"""
function loglikelihood(θz, θy, data::neuraldata)

    @unpack spikes, input_data = data
    @unpack dt = input_data
    λ, = rand(θz,θy,input_data)
    sum(logpdf.(Poisson.(vcat(λ...)*dt), vcat(spikes...)))

end


"""
"""
function regress_init(data::Dict, f_str::String)

    ΔLR = map((T,L,R)-> diffLR(T,L,R, data["dt"]), data["nT"], data["leftbups"], data["rightbups"])
    SC_byneuron = group_by_neuron(data["spike_counts"], data["ntrials"], data["N"])
    λ0_byneuron = group_by_neuron(data["λ0"], data["ntrials"], data["N"])

    py = map(k-> compute_p0(ΔLR, k, data["dt"], f_str), SC_byneuron)

end


"""
"""
function compute_p0(ΔLR,k,dt,f_str;nconds::Int=7)

    #conds_bins, = qcut(vcat(ΔLR...),nconds,labels=false,duplicates="drop",retbins=true)
    conds_bins = encode(LinearDiscretizer(binedges(DiscretizeUniformWidth(nconds), vcat(ΔLR...))), vcat(ΔLR...))

    fr = map(i -> (1/dt)*mean(vcat(k...)[conds_bins .== i]),1:nconds)

    A = vcat(ΔLR...)
    b = vcat(k...)
    c = hcat(ones(size(A, 1)), A) \ b

    if f_str == "exp"
        p = vcat(minimum(fr),c[2])
    elseif (f_str == "sig")
        p = vcat(minimum(fr),maximum(fr)-minimum(fr),c[2],0.)
    elseif f_str == "softplus"
        p = vcat(minimum(fr),(1/dt)*c[2],0.)
    end

    #added because was getting log problem later, since rate function canot be negative
    p[1] == 0. ? p[1] += eps() : nothing

    return p

end
