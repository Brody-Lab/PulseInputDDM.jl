

#################################### Choice observation model #################################

function compute_Hessian(pz, pd, data; n::Int=53) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], pd["final"], fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, pz["lb"], pz["ub"], n=n)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end


function compute_H_CI!(pz, pd, data; n::Int=53)
    
    H = compute_Hessian(pz, pd, data; n=n)

    #badindices = findall(abs.(vcat(pz[:final],pd[:final])[vcat(pz[:fit],pd[:fit])]) 
    #    .< 1e-4)

    #gooddims = setdiff(1:size(H,1),badindices)
    
    gooddims = 1:size(H,1)
    
    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(copy(pz["final"]), copy(pd["final"]), 
        fit_vec, data["dt"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));

    pz["CI_plus"], pd["CI_plus"] = map_split_combine(p_opt + CI, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    pz["CI_minus"], pd["CI_minus"] = map_split_combine(p_opt - CI, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    return pz, pd
    
end

function load_and_optimize(path::String, sessids, ratnames; dt::Float64=1e-2, n::Int=53,
        show_trace::Bool=true,
        fit_initial::Vector{Float64}=vcat(1e-6,20.,-0.1,100.,5.,0.2,0.005),
        fit_latent::BitArray{1}=trues(dimz),
        lb::Vector{Float64}=[eps(), 4., -5., eps(), eps(), eps(), eps()],
        ub::Vector{Float64}=[10., 100, 5., 800., 40., 2., 10.])
    
    data = aggregate_choice_data(path,sessids,ratnames,dt)
    
    #parameters of the latent model
    pz = Dict("name" => vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        "fit" => fit_latent,
        "initial" => fit_initial,
        "lb" => lb,
        "ub" => ub)

    #parameters for the choice observation
    pd = Dict("name" => vcat("bias","lapse"), "fit" => trues(2), 
        "initial" => vcat(0.,0.5))
    
    pz, pd = optimize_model(pz, pd, data; n=n, show_trace=show_trace)
    
    return pz, pd
    
end

function load_and_optimize(data; n::Int=53,
        show_trace::Bool=true,
        fit_latent::BitArray{1}=trues(dimz),
        fit_initial::Vector{Float64}=vcat(1e-6,20.,-0.1,100.,5.,0.2,0.005),
        lb::Vector{Float64}=[eps(), 4., -5., eps(), eps(), eps(), eps()],
        ub::Vector{Float64}=[10., 100, 5., 800., 40., 2., 10.])
    
    #parameters of the latent model
    pz = Dict("name" => vcat("σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"),
        "fit" => fit_latent,
        "initial" => fit_initial,
        "lb" => lb,
        "ub" => ub)

    #parameters for the choice observation
    pd = Dict("name" => vcat("bias","lapse"), "fit" => trues(2), 
        "initial" => vcat(0.,0.5))
    
    pz, pd = optimize_model(pz, pd, data; n=n, show_trace=show_trace)
    
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
    ll(x) = ll_wrapper(x, p_const, fit_vec, data, pz["lb"], pz["ub"], n=n)
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz["state"], pd["state"] = map_split_combine(p_opt, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    pz["final"], pd["final"] = pz["state"], pd["state"]
    
    return pz, pd, opt_output, state
        
end

function ll_wrapper(p_opt::Vector{TT}, p_const::Vector{Float64},
        fit_vec::Union{BitArray{1},Vector{Bool}}, 
        data::Dict, lb::Vector{Float64}, 
        ub::Vector{Float64}; n::Int=53) where {TT <: Any}
          
    pz, pd = map_split_combine(p_opt, p_const, fit_vec, data["dt"], lb, ub)
    
    LL = compute_LL(pz, pd, data; n=n)
    
    return -LL
              
end

"""
    compute_LL(pz,bias,data;n::Int=53)

    compute LL for choice observation model

"""
compute_LL(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials(pz, pd, data, n=n))
