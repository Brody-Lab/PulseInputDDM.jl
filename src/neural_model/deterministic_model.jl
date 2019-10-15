############ Deterministic latent variable model (only observation noise) ############

function compute_H_CI!(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String)
    
    H = compute_Hessian(pz, py, data, f_str)
        
    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);
    
    try
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));
    catch
        @warn "CI computation failed."
    end

    pz["CI_plus"], py["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], py["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #if generative parameters exist, identify which ones have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(py, "generative")
        py["within_CI"] = map((x,y,z)-> [((x .< z) .& (y .> z)) for (x,y,z) in zip(x,y,z)], 
            py["CI_minus"], py["CI_plus"], py["generative"])
    end
    
    return pz, py
    
end

"""
    compute_Hessian(pz,py,data;n::Int=53, f_str="softplus")

    compute Hessian

"""
function compute_Hessian(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String)
    
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], py["final"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str)
    
    return H = ForwardDiff.hessian(ll, p_opt)
        
end

function optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String; 
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        extended_trace::Bool=false) where {TT <: Any}
    
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])
    
    pz = check_pz!(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    p_opt, p_const = split_combine_invmap(pz["state"], py["state"], fit_vec, data[1]["dt"], f_str, pz["lb"], pz["ub"])

    ###########################################################################################
    ## Optimize
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data[1]["dt"], 
        f_str, py["N"], py["dimy"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str)
    
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace, extended_trace= extended_trace);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
        
    return pz, py, opt_output, state
    
end

function optimize_model_con(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String; 
        x_tol::Float64=1e-16, f_tol::Float64=1e-16, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e3)) where {TT <: Any}
    
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])
    
    pz = check_pz!(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz["state"], py["state"]),fit_vec)
    
    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str)
    
    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace);
    p_opt = Optim.minimizer(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
    
    return pz, py, opt_output
    
end

function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)   
    LL = compute_LL(pz, py, data, f_str)
        
    return -LL
              
end

function compute_LL(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}},f_str::String) where {T <: Any}
    
    LL = sum(map((py,data)-> sum(LL_all_trials(pz, py, data, f_str)), py, data))
            
end

function LL_all_trials(pz::Vector{TT}, py::Vector{Vector{TT}}, data::Dict, 
        f_str::String) where {TT <: Any}
     
    dt = data["dt"]
    use_bin_center = data["use_bin_center"]
    #trials = sample(1:data["ntrials"], min(100,data["ntrials"]), replace = false)
    trials = sample(1:data["ntrials"], data["ntrials"], replace = false)
    
    sum(pmap((L,R,nT,nL,nR,k,λ0)-> LL_single_trial(pz,py,L,R,nT,nL,nR,k,λ0,dt,f_str,use_bin_center), 
        data["leftbups"][trials], data["rightbups"][trials], data["nT"][trials], 
            data["binned_leftbups"][trials], 
        data["binned_rightbups"][trials], data["spike_counts"][trials], 
            data["λ0"][trials], batch_size=length(trials)))
        
end

function LL_single_trial(pz::Vector{TT}, py::Vector{Vector{TT}},
        L::Vector{Float64}, R::Vector{Float64}, nT::Int, 
        nL::Vector{Int}, nR::Vector{Int},
        k::Vector{Vector{Int}}, λ0::Vector{Vector{Float64}}, dt::Float64, 
        f_str::String,use_bin_center::Bool) where {TT <: Any}
    
    a = sample_latent(nT,L,R,nL,nR,pz,use_bin_center;dt=dt)
    #sum(map((py,k,λ0)-> sum(poiss_LL.(k, map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0), dt)), py, k, λ0))
    sum(map((py,k,λ0)-> sum(logpdf.(Poisson.(map((a, λ0)-> f_py!(a, λ0, py, f_str), a, λ0) * dt), k)), 
            py, k, λ0))
    
end

######### regression initialization ##############################################################

function regress_init(data::Dict, f_str::String)
    
    ΔLR = map((T,L,R)-> diffLR(T,L,R, data["dt"]), data["nT"], data["leftbups"], data["rightbups"])   
    SC_byneuron = group_by_neuron(data["spike_counts"], data["ntrials"], data["N"])
    λ0_byneuron = group_by_neuron(data["λ0"], data["ntrials"], data["N"])

    py = map(k-> compute_p0(ΔLR, k, data["dt"], f_str), SC_byneuron) 
        
end

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