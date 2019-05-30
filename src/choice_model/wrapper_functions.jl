#################################### dx Choice observation model #################################

function generate_syn_data_fit_CI_dx(pz::Dict{}, pd::Dict{}, ntrials::Int; 
        dx::Float64=0.25, dt=1e-2)
   
    data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials)
    
    data = bin_clicks!(data;dt=dt)
    
    pz, pd, = optimize_model_dx(pz, pd, data; dx=dx, show_trace=true)   
    pz, pd = compute_H_CI_dx!(pz, pd, data; dx=dx)

end

function compute_Hessian_dx(pz, pd, data; dx::Float64=0.25) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(pz["final"], pd["final"], fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_dx(x, data, parameter_map_f, dx=dx)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end


function compute_H_CI_dx!(pz, pd, data; dx::Float64=0.25)
    
    H = compute_Hessian_dx(pz, pd, data; dx=dx)
    
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
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(pd, "generative")
        pd["within_CI"] = (pd["CI_minus"] .< pd["generative"]) .& (pd["CI_plus"] .> pd["generative"])
    end
    
    return pz, pd
    
end

function optimize_model_dx(pz::Dict{}, pd::Dict{}, data; dx::Float64=0.25,
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
        
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])
    
    pz = check_pz!(pz)
            
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap(pz["state"], pd["state"], fit_vec, data["dt"], pz["lb"], pz["ub"])

    ###########################################################################################
    ## Optimize
    parameter_map_f(x) = map_split_combine(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_dx(x, data, parameter_map_f, dx=dx)
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    
    return pz, pd, opt_output, state
        
end

function ll_wrapper_dx(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        dx::Float64=0.25) where {TT <: Any}
          
    pz, pd = parameter_map_f(p_opt)
    
    LL = compute_LL_dx(pz, pd, data; dx=dx)
    
    return -LL
              
end

compute_LL_dx(pz::Vector{T}, pd::Vector{T}, data; dx::Float64=0.25) where {T <: Any} = sum(LL_all_trials_dx(pz, pd, data, dx=dx))

function LL_all_trials_dx(pz::Vector{TT}, pd::Vector{TT}, data::Dict; dx::Float64=0.25) where {TT}
        
    bias,lapse = pd[1],pd[2]
    dt = data["dt"]
    P,M,xc,n,xe = initialize_latent_model_dx(pz, dx, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    nbinsL, Sfrac = bias_bin(bias,xe,dx,n)
            
    output = pmap((L,R,nT,nL,nR,choice) -> LL_single_trial(pz, P, M, dx, xc,
        L, R, nT, nL, nR, nbinsL, Sfrac, choice, n, dt),
        data["leftbups"], data["rightbups"], data["nT"], data["binned_leftbups"], 
        data["binned_rightbups"],data["pokedR"])   
    
end

function initialize_latent_model_dx(pz::Vector{TT}, dx::Float64, dt::Float64; 
        L_lapse::UU=0., R_lapse::UU=0.) where {TT,UU}
    
    vari,B,lambda,vara = pz[1:4]                      #break up latent variables
   
    xc,n,xe = bins_dx(B,dx)                              # spatial bin centers, width and edges
    
    P = P0(vari,n,dx,xc,dt; 
        L_lapse=L_lapse, R_lapse=R_lapse)             # make initial latent distribution
   
    M = zeros(TT,n,n)                                 # build empty transition matrix
    M!(M,vara*dt,lambda,zero(TT),dx,xc,n,dt)          # build state transition matrix for no input time bins
    
    return P, M, xc, n, xe
    
end

#convert(::Type{T}, x::ForwardDiff.Dual) where {T<:Number} = T(x.value)

function bins_dx(B::TT,dx::Float64) where {TT}
    
    #dx = 2. *B/(n-2);  #bin width
    n2 = ceil(B/dx)
    
    #n = 2*Int(ceil(B/dx)) + 1
    
    n = 2*Int(n2) + 1
    
    #xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,
    #    collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2)))); #centers
           
    xc = collect(-n2:n2)*dx
    #n2*dx == B ? (xc[end] = B + dx; xc[1] = -(B + dx)) : (xc[end] = 2*B - (n2-1)*dx; xc[1] = -(2*B - (n2-1)*dx))
    n2*dx == B ? xc[[end,1]] = [convert(eltype(xc),B) + dx, -(convert(eltype(xc),B) + dx)] : 
        xc[[end,1]] = [2*convert(eltype(xc),B) - (n2-1)*dx, -(2*convert(eltype(xc),B) - (n2-1)*dx)]
    
    #this will need to be fixed, I think, but only for the choice model
    xe = cat(xc[1] - dx/2,xc .+ dx/2, dims=1) #edges 
    
    return xc, n, xe
    
end


#################################### Choice observation model #################################

function generate_syn_data_fit_CI(pz::Dict{}, pd::Dict{}, ntrials::Int; 
        n::Int=53, dt=1e-2)
   
    data = sample_inputs_and_choices(pz["generative"], pd["generative"], ntrials)
    
    data = bin_clicks!(data;dt=dt)
    
    pz, pd, = optimize_model(pz, pd, data; n=n, show_trace=true)   
    pz, pd = compute_H_CI!(pz, pd, data; n=n)

end

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
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(pd, "generative")
        pd["within_CI"] = (pd["CI_minus"] .< pd["generative"]) .& (pd["CI_plus"] .> pd["generative"])
    end
    
    return pz, pd
    
end

function load_and_optimize(path::String, sessids, ratnames; n::Int=53, dt::Float64=1e-2,
        pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => trues(dimz),
        "initial" => [1e-6,20.,-0.1,100.,5.,0.2,0.005],
        "lb" => [eps(), 4., -5., eps(), eps(), eps(), eps()],
        "ub" => [10., 100, 5., 800., 40., 2., 10.]),
        show_trace::Bool=true, iterations::Int=Int(2e3),
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2)
    
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
        show_trace::Bool=true, iterations::Int=Int(2e3),
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2)

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
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
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


#################################### New Choice model #################################

function compute_Hessian_new(pz, pd, data; n::Int=53) where {TT <: Any}
    
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(pz["final"], pd["final"], fit_vec, data["dt"], pz["lb"], pz["ub"])
    
    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_new(x, data, parameter_map_f, n=n)

    return H = ForwardDiff.hessian(ll, p_opt);
        
end


function compute_H_CI_new!(pz, pd, data; n::Int=53)
    
    H = compute_Hessian_new(pz, pd, data; n=n)
    
    gooddims = 1:size(H,1)
    
    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(copy(pz["final"]), copy(pd["final"]), 
        fit_vec, data["dt"], pz["lb"], pz["ub"])

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))
    
    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)
    
    #identify which ML parameters have generative parameters within the CI 
    if haskey(pz, "generative")
        pz["within_CI"] = (pz["CI_minus"] .< pz["generative"]) .& (pz["CI_plus"] .> pz["generative"])
    end
    
    if haskey(pd, "generative")
        pd["within_CI"] = (pd["CI_minus"] .< pd["generative"]) .& (pd["CI_plus"] .> pd["generative"])
    end
    
    return pz, pd
    
end


function optimize_model_new(pz::Dict{}, pd::Dict{}, data; n=53,
        x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2,
        iterations::Int=Int(2e3), show_trace::Bool=true)
        
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])
    
    pz = check_pz!(pz)
            
    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_combine_invmap_new(pz["state"], pd["state"], fit_vec, data["dt"], pz["lb"], pz["ub"])

    parameter_map_f(x) = map_split_combine_new(x, p_const, fit_vec, data["dt"], pz["lb"], pz["ub"])
    ll(x) = ll_wrapper_new(x, data, parameter_map_f, n=n)
    opt_output, state = opt_ll(p_opt, ll; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, iterations=iterations,
        show_trace=show_trace)
    p_opt = Optim.minimizer(opt_output)
    
    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    
    return pz, pd, opt_output, state
        
end

function ll_wrapper_new(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        n::Int=53) where {TT <: Any}
          
    pz, pd = parameter_map_f(p_opt)
    
    LL = compute_LL_new(pz, pd, data; n=n)
    
    return -LL
              
end

compute_LL_new(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials_new(pz, pd, data, n=n))
