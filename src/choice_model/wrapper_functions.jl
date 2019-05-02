

#################################### Choice observation model #################################

function compute_Hessian(pz, pd, data; n::Int=53) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], pd["final"], fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, n=n)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end


function compute_H_CI!(pz, pd, data; n::Int=53)
    
    H = compute_Hessian(pz, pd, data; n=n)
    
    gooddims = 1:size(H,1)
    
    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(copy(pz["final"]), copy(pd["final"]), 
        fit_vec, data["dt"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))
    
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #identify which ML parameters have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_bounds"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(pd, "generative")
        pd["within_bounds"] = (pd["CI_minus"] .< pd["generative"]) .& (pd["CI_plus"] .> pd["generative"])
    end
    
    return pz, pd
    
end

function load_and_optimize(path::String, sessids, ratnames; n::Int=53, dt::Float64=1e-2,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3))
    
    data = aggregate_choice_data(path,sessids,ratnames)
    data = bin_clicks!(data;dt=dt)
        
    pz, pd = load_and_optimize(data; n=n,
        pz=pz, show_trace=show_trace, iterations=iterations)
    
    return pz, pd
    
end

function load_and_optimize(data; n::Int=53,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3))

    #parameters for the choice observation
    pd = Dict("name" => vcat("bias","lapse"), "fit" => trues(2), 
        "initial" => vcat(0.,0.5))
    
    pz, pd = optimize_model(pz, pd, data; n=n, show_trace=show_trace, iterations=iterations)
    
    return pz, pd
    
end

"""
    optimize_model(pz, pd, data; n, x_tol, f_tol, g_tol, iterations)

    Optimize parameters specified within fit vectors.

"""
function optimize_model(pz::Dict{}, pd::Dict{}, data; n=53,
        x_tol::Float64=1e-4, f_tol::Float64=1e-6, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
        
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])
    
    pz = check_pz!(pz)
            
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(pz["state"], pd["state"], fit_vec, data["dt"], pz["lb"], pz["ub"])

    ###########################################################################################
    ## Optimize
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper(x, data, parameter_map_f, n=n)
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    
    return pz, pd, opt_output, state
        
end

function ll_wrapper(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        n::Int=53) where {TT <: Any}
          
    pz, pd = parameter_map_f(p_opt)
    
    LL = compute_LL(pz, pd, data; n=n)
    
    return -LL
              
end

"""
    compute_LL(pz,bias,data;n::Int=53)

    compute LL for choice observation model

"""
compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials(pz, pd, data, n=n))
