"""
"""
@flattenable @with_kw struct θneural{T1, T2} <: DDMθ
    θz::T1 = θz() | true
    θy::T2 | true
    N::Vector{Int} | false
end


"""
"""
@with_kw struct Sigmoid{T1}
    a::T1=10.
    b::T1=10.
    c::T1=1.
    d::T1=0.
end

function (θ::Sigmoid)(x::Union{U,Vector{U}}, λ0::Union{Float64,Vector{Float64}}) where U <: Real

    @unpack a,b,c,d = θ

    y = c * x + d
    y = a + b * logistic!(y)
    y = softplus(y + λ0)

end

@with_kw struct Softplus{T1}
    a::T = 10.
    c::T = 5.0*rand([-1,1])
    d::T = 0
end

function (θ::Softplus)(x::Union{U,Vector{U}}, λ0::Union{Float64,Vector{Float64}}) where U <: Real

    @unpack a,c,d = θ

    y = a + softplus(c*x + d)
    y = max(eps(),y + λ0)

end


"""
"""
@with_kw struct θy{T1}
    θ::T1
end


@with_kw struct neuraldata{T1,T2,T3} <: DDMdata
    input_data::T1
    spikes::T2
    N::T3
end

@with_kw struct neuralDDM{T,U} <: DDM
    θ::T = θneural()
    data::U
end


"""
"""
function unflatten(x::Vector{T}, dims::Vector{Int}, f::String) where {T <: Real}

    dims2 = vcat(0,cumsum(dims))
    θy = map(idx-> collect(partition(x[dimz+1:end], 4))[idx], [dims2[i]+1:dims2[i+1] for i in 1:length(dims2)-1])
    θneural(θz(Tuple(x[1:dimz])...), θy, f, dims)

end


"""
    loglikelihood(x, data; n=53)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input. Used
in optimization, Hessian and gradient computation.
"""
function loglikelihood(x::Vector{T}, data, dims::Vector{Int}, f::String; n::Int=53) where {T <: Real}

    θ = unflatten(x,dims,f)
    loglikelihood(θ, data; n=n)

end


"""
    gradient(model; n=53)
"""
function gradient(model::neuralDDM; n::Int=53)

    @unpack θ, data = model
    @unpack N, f = θ
    x = flatten(θ)
    #x = [flatten(θ)...]
    ℓℓ(x) = -loglikelihood(x, data, N, f; n=n)

    ForwardDiff.gradient(ℓℓ, x)

end


"""
    flatten(θ)

Extract parameters related to the choice model from a struct and returns an ordered vector
```
"""
function flatten(θ::θneural)

    @unpack θy, θz = θ
    @unpack σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = θz
    vcat(σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ, vcat(vcat(θy...)...))

end


"""
    default_parameters(f_str, cells_per_session, num_sessions;generative=false)

Returns two dictionaries of default model parameters.
"""
function default_parameters(f_str::String, cells_per_session::Vector{Int},
        num_sessions::Int; generative::Bool=false)

    pz::Dict = Dict("name" => ["σ_i","B", "λ", "σ_a","σ_s","ϕ","τ_ϕ"],
        "fit" => vcat(trues(4),falses(3)),
        "initial" => vcat(0.1, 11., -0.1, 20., 0.1, 0.8, 0.01),
        "lb" => [eps(), 8., -5., eps(), eps(), 0.01, 0.005],
        "ub" => [100., 100., 5., 100., 2.5, 1.2, 1.5])

    if f_str == "softplus"

        dimy = 3

        py = Dict("dimy"=> dimy,
            "cells_per_session"=> cells_per_session,
            "num_sessions"=> num_sessions,
            "fit" => [[trues(dimy) for n in 1:N] for N in cells_per_session],
            "initial" => [[Vector{Float64}(undef,dimy) for n in 1:N] for N in cells_per_session],
            "lb" => [[[eps(),-10.,-10.] for n in 1:N] for N in cells_per_session],
            "ub" => [[[100.,10.,10.] for n in 1:N] for N in cells_per_session])

    elseif f_str == "sig"

        dimy = 4

        py = Dict("dimy"=> dimy,
            "cells_per_session"=> cells_per_session,
            "num_sessions"=> num_sessions,
            "fit" => [[trues(dimy) for n in 1:N] for N in cells_per_session],
            "initial" => [[Vector{Float64}(undef,dimy) for n in 1:N] for N in cells_per_session],
            "lb" => [[[-100.,0.,-10.,-10.] for n in 1:N] for N in cells_per_session],
            "ub" => [[[100.,100.,10.,10.] for n in 1:N] for N in cells_per_session])
    end

    if generative

        pz["generative"] = [0.5, 15., -0.5, 10., 1.2, 0.6, 0.02]
        pz["initial"][.!pz["fit"]] = pz["generative"][.!pz["fit"]]

        if f_str == "softplus"
            py["generative"] = [[[10., 5.0*sign(randn()), 0.] for n in 1:N] for N in cells_per_session]
        elseif f_str == "sig"
            py["generative"] = [[[10., 10., 1., 0.] for n in 1:N] for N in cells_per_session]
        end
    end

    return pz, py

end


"""
    default_parameters(data, f_str; generative=false)

Returns two dictionaries of default model parameters, using the data to initialize py.
"""
function default_parameters(data, f_str::String; generative::Bool=false, show_trace::Bool=false)

    num_sessions, cells_per_session = length(data), [data["N"] for data in data]
    pz, py = default_parameters(f_str, cells_per_session, num_sessions; generative=false)
    py = initialize_py!(pz, py, data, f_str; show_trace=show_trace)

    return pz, py

end



"""
"""
function initialize_py!(pz, py, data, f_str; show_trace::Bool=false)

    pztemp = deepcopy(pz)
    pztemp["fit"] = falses(dimz)
    pztemp["initial"][[1,4,5]] .= 2*eps()

    py["initial"] = map(data-> regress_init(data, f_str), data)
    pztemp, py = optimize_model_deterministic(pztemp, py, data, f_str, show_trace=show_trace)
    delete!(py,"final")

    return py

end


"""
    optimize_model(f_str; num_sessions, num_trials_per_session, cells_per_session;
        n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1),
        generative=true, rng=1, dt=1e-2, use_bin_center=true)

Generate parameters and data and thn Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries, and specification of which parameters to fit.
"""
function optimize_model(f_str::String, num_sessions::Int,
        num_trials_per_session::Vector{Int},
        cells_per_session::Vector{Int};
        n::Int=53, x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e1), generative::Bool=true,
        rng::Int=1, dt::Float64=1e-2, use_bin_center::Bool=true)

    pz, py, data = default_parameters_and_data(f_str, num_sessions, num_trials_per_session,
        cells_per_session; generative=true, rng=rng, dt=dt, use_bin_center=true)

    py = initialize_py!(pz, py, data, f_str; show_trace=false)

    pz, py, converged, opt_output = optimize_model(pz, py, data, f_str;
        n=n, x_tol=x_tol, f_tol=f_tol, g_tol=g_tol,
        iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    return pz, py, data, converged, opt_output

end


"""
    optimize_model(data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

BACK IN THE DAY, TOLS USED TO BE x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize_model(data::Vector{Dict{Any,Any}}, f_str::String;
        n::Int=53, x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e1))

    pz, py = default_parameters(data, f_str; show_trace=false, generative=false)

    pz, py, converged, opt_output = optimize_model(pz, py, data, f_str;
        n=n, x_tol=x_tol, f_tol=f_tol, g_tol=g_tol,
        iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations)

    return pz, py, converged, opt_output

end


"""
    optimize_model(pz, py, data, f_str; n=53, x_tol=1e-10,
        f_tol=1e-6, g_tol=1e-3,iterations=Int(2e3), show_trace=true,
        outer_iterations=Int(2e3), outer_iterations=Int(2e1))

Optimize model parameters. pz and py are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize_model(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        n::Int=53, x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true,
        outer_iterations::Int=Int(2e1)) where {TT <: Any}

    println("optimize! \n")

    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(py,"state") ? nothing : py["state"] = deepcopy(py["initial"])

    check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])
    lb = combine_latent_and_observation(pz["lb"], py["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], py["ub"])[fit_vec]

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,py,data,f_str,n; state="state")

    opt_output = opt_func(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations=iterations, show_trace=show_trace, outer_iterations=outer_iterations);
    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], py["state"] = parameter_map_f(p_opt)
    pz["final"], py["final"] = pz["state"], py["state"]

    println("optimization complete \n")
    println("converged: $converged \n")

    return pz, py, converged, opt_output

end


"""
    ll_wrapper(p_opt, data, parameter_map_f, f_str, n)

A wrapper function that accepts a vector of mixed parameters, splits the vector
into two vectors based on the parameter mapping function provided as an input,
and compute the negative log likelihood of the data given the parametes. Used
in optimization.
"""
function ll_wrapper(p_opt::Vector{TT}, data::Vector{Dict{Any,Any}}, parameter_map_f::Function, f_str::String,
        n::Int) where {TT <: Any}

    pz, py = parameter_map_f(p_opt)
    -compute_LL(pz, py, data, f_str, n)

end


"""
"""
function split_opt_params_and_close(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String,
        n::Int; state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])

    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], py[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec),
        py["cells_per_session"], py["dimy"])

    ll(x) = ll_wrapper(x, data, parameter_map_f, f_str, n)

    return p_opt, ll, parameter_map_f

end


"""
    compute_CIs!(pz, py, data, f_str; n=53)

compute LL for your model. returns a scalar
"""
function compute_CIs!(pz::Dict{}, py::Dict{}, H, f_str::String; n::Int=53)

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

    p_opt, ll, parameter_map_f = split_opt_params_and_close(pz,py,[Dict()],f_str,n; state="final")

    pz["CI_plus_hessian"], py["CI_plus_hessian"] = parameter_map_f(p_opt + CI)
    pz["CI_minus_hessian"], py["CI_minus_hessian"] = parameter_map_f(p_opt - CI)

    return pz, py

end


"""
    compute_Hessian(pz, py, data, f_str; n=53)

compute Hessian
"""
function compute_Hessian(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        n::Int=53, state::String="state")

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close(pz,py,data,f_str,n; state=state)
    ForwardDiff.hessian(ll, p_opt)

end


"""
"""
function compute_gradient(pz::Dict{}, py::Dict{}, data::Vector{Dict{Any,Any}}, f_str::String;
        n::Int=53, state::String="state")

    println("computing gradient \n")
    p_opt, ll, = split_opt_params_and_close(pz,py,data,f_str,n; state=state)
    ForwardDiff.gradient(ll, p_opt)

end


"""
    combine_latent_and_observation(pz,py)

Combines two vectors into one vector. The first vector is for components related
to the latent variables, the second vector is for components related to the observation model.
### Examples
```jldoctest
julia> pz, py = pulse_input_DDM.default_parameters();

julia> p = pulse_input_DDM.combine_latent_and_observation(pz["initial"], py["initial"]);

julia> pulse_input_DDM.split_latent_and_observation(p) == (pz["initial"], py["initial"])
true
```
"""
combine_latent_and_observation(pz::Union{Vector{TT},BitArray{1}},
    py::Union{Vector{Vector{Vector{TT}}}, Vector{Vector{BitArray{1}}}}) where {TT <: Any} = vcat(pz,vcat(vcat(py...)...))



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


"""
    LL_across_range(pz, py, data)

"""
function LL_across_range(pz::Dict, py::Dict, data, f_str, lb, ub, i; n::Int=53, state::String="final")

    fit_vec = combine_latent_and_observation(pz["fit"], py["fit"])

    lb_vec = combine_latent_and_observation(lb[1], lb[2])
    ub_vec = combine_latent_and_observation(ub[1], ub[2])

    ll_θ = compute_LL(pz[state], py[state], data, f_str, n)

    fit_vec2 = falses(length(fit_vec))
    fit_vec2[i] = true

    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], py[state]), fit_vec2)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec2),
        py["cells_per_session"], py["dimy"])

    ll(x) = -ll_wrapper([x], data, parameter_map_f, f_str, n) - (ll_θ - 1.92)

    xs = range(lb_vec[i], stop=ub_vec[i], length=50)
    LLs = map(x->ll(x), xs)

    return LLs, xs

end
