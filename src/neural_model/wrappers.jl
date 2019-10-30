"""
    default_parameters(f_str, cells_per_session, num_sessions;generative=false)

Returns two dictionaries of default model parameters.
"""
function default_parameters(f_str::String, cells_per_session::Vector{Int}, 
        num_sessions::Int; generative::Bool=false)
    
    if f_str == "softplus"
        
        dimy = 3
        
        py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), cells_per_session),
            "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in cells_per_session],
            "dimy"=> dimy,
            "N"=> cells_per_session,
            "nsessions"=> num_sessions,
            "lb" => [[[2*eps(),-Inf,-Inf] for n in 1:N] for N in cells_per_session],
            "ub" => [[[Inf,Inf,Inf] for n in 1:N] for N in cells_per_session])
        
    elseif f_str == "sig"
        
        dimy = 4
        
        py = Dict("fit" => map(N-> repeat([trues(dimy)],outer=N), cells_per_session),
            "initial" => [[[Vector{Float64}(undef,dimy)] for n in 1:N] for N in cells_per_session],
            "dimy"=> dimy,
            "N"=> cells_per_session,
            "nsessions"=> num_sessions,
            "lb" => [[[-100.,0.,-10.,-10.] for n in 1:N] for N in cells_per_session],
            "ub" => [[[100.,100.,10.,10.] for n in 1:N] for N in cells_per_session])
    end  
        
    pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => vcat(falses(1),trues(2),falses(4)),
        "initial" => [2*eps(), 10., -0.1, 2*eps(), 2*eps(), 1., 0.01],
        "lb" => [eps(), 8., -5., eps(), eps(), 0.01, 0.005],
        "ub" => [100., 100., 5., 100., 2.5, 1.2, 1.5])

    if generative
        pz["generative"] = [eps(), 18., -0.5, 5., 1.5, 0.4, 0.02]
        pd["generative"] = [1.,0.05]
    end

    return pz, py

end


"""
    compute_CIs!(pz, py, data, f_str; dx=0.25)

compute LL for your model. returns a scalar
"""
function compute_CIs!(pz::Dict{}, py::Dict{}, H, f_str::String; dx::Float64=0.25)

    println("computing confidence intervals \n")
    
    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8)
    
    try
        gooddims = 1:size(H,1)
        evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
        otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
        gooddims = setdiff(gooddims,otherbad)
        CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])));
    catch
        @warn "CI computation failed."
    end
    
    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,py,Dict(); state="final")

    pz["CI_plus_hessian"], py["CI_plus_hessian"] = parameter_map_f(p_opt + CI)
    pz["CI_minus_hessian"], py["CI_minus_hessian"] = parameter_map_f(p_opt - CI)

    return pz, py

end


"""
    compute_Hessian(pz, py, data, f_str; dx=0.25)

compute Hessian
"""
function compute_Hessian(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        dx::Float64=0.25, state::String="state")

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close(pz,py,data; dx=dx, state=state)
    ForwardDiff.hessian(ll, p_opt)

end


"""
    optimize_model(pz, py, data, f_str; dx=0.25, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        dx::Float64=0.25; x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e1)) where {TT <: Any}

    println("optimize! \n")

    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])

    check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]
    
    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,py,data; dx=dx, state="state")
    
    p_opt[p_opt .< lb] .= lb[p_opt .< lb]
    p_opt[p_opt .> ub] .= ub[p_opt .> ub]

    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace, outer_iterations=outer_iterations);
    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]
    
    println("optimization complete \n")
    println("converged: $converged \n")

    return pz, py, converged

end


"""
    ll_wrapper(p_opt, data, parameter_map_f, f_str; dx=0.25)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input,
and compute the negative log likelihood of the data given the parametes. Used
in optimization.
"""
function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String;
        dx::Float64=0.25) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)
    -compute_LL(pz, py, data, f_str; dx=dx)

end


"""
    compute_LL(pz, py, data, n, f_str)

compute LL for your model. returns a scalar
"""
function compute_LL(pz::Vector{T}, py::Vector{Vector{Vector{T}}}, data::Vector{Dict{Any,Any}},
        f_str::String; dx::Float64=0.25) where {T <: Any}

    LL = sum(map((py,data)-> sum(LL_all_trials(pz, py, data, f_str; dx=dx)), py, data))

end


"""
"""
function split_opt_params_and_close(pz::Dict{}, py::Dict{}, data::Dict{}; dx::Float64=0.25, state::String="state")
    
    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], py[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec), py["N"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str; dx=dx)
    
    return p_opt, ll, parameter_map_f

end


"""
"""
function compute_gradient(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        dx::Float64=0.25)

    println("computing gradient \n")
    p_opt, ll, = split_opt_params_and_close(pz,py,data; dx=dx, state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    combine_latent_and_observation(pz,py)

Combines two vectors into one vector. The first vector is for components related
to the latent variables, the second vector is for components related to the observation model.
### Examples
```jldoctest
julia> pz, pd = pulse_input_DDM.default_parameters();

julia> p = pulse_input_DDM.combine_latent_and_observation(pz["initial"], py["initial"]);

julia> pulse_input_DDM.split_latent_and_observation(p) == (pz["initial"], py["initial"])
true
```
"""
combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}}, 
    py::Union{Vector{Vector{Vector{TT}}},Vector{Vector{BitArray{1}}}}) where {TT <: Any} = vcat(pz,vcat(vcat(py...)...))



"""
    split_latent_and_observation(p)

Splits a vector up into two vectors. The first vector is for components related
to the latent variables, the second is for components related to the observation model.
### Examples
```jldoctest
julia> pz, py = pulse_input_DDM.default_parameters();

julia> p = pulse_input_DDM.combine_latent_and_observation(pz["initial"], py["initial"]);

julia> pulse_input_DDM.split_latent_and_observation(p) == (pz["initial"], py["initial"])
true
```
"""
function split_latent_and_observation(p::Vector{T}, N::Vector{Int}, dimy::Int) where {T <: Any}
                
    pz = p[1:dimz]
    #linear index that defines the beginning of a session
    iter = cumsum(vcat(0,N))*dimy
    #group single session parameters into 2D arrays
    py = map(i-> reshape(p[dimz+iter[i-1]+1:dimz+iter[i]], dimy, N[i-1]), 2:length(iter))
    #break up single session 2D arrays into an array of arrays
    py = map(i-> map(j-> py[i][:,j], 1:N[i]), 1:length(N))

    return pz, py
    
end
