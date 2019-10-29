"""
    optimize_model_n(pz, pd, data; n=53, x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true, outer_iterations=Int(1e1))

Optimize model parameters. pz and pd are dictionaries that contains initial values, boundaries,
and specification of which parameters to fit.
"""
function optimize_model_n(pz::Dict{}, pd::Dict{}, data::Dict{}; n=53,
        x_tol::Float64=1e-10, f_tol::Float64=1e-6, g_tol::Float64=1e-3,
        iterations::Int=Int(2e3), show_trace::Bool=true, 
        outer_iterations::Int=Int(1e1))

    println("optimize! \n")
    haskey(pz,"state") ? nothing : pz["state"] = deepcopy(pz["initial"])
    haskey(pd,"state") ? nothing : pd["state"] = deepcopy(pd["initial"])

    check_pz(pz)

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    lb = combine_latent_and_observation(pz["lb"], pd["lb"])[fit_vec]
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])[fit_vec]

    p_opt, ll, parameter_map_f = split_opt_params_and_close_n(pz,pd,data; n=n, state="state")

    p_opt[p_opt .< lb] .= lb[p_opt .< lb]
    p_opt[p_opt .> ub] .= ub[p_opt .> ub]

    opt_output = opt_func_fminbox(p_opt, ll, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, outer_iterations=outer_iterations, show_trace=show_trace)

    p_opt, converged = Optim.minimizer(opt_output), Optim.converged(opt_output)

    pz["state"], pd["state"] = parameter_map_f(p_opt)
    pz["final"], pd["final"] = pz["state"], pd["state"]
    println("optimization complete \n")
    println("converged: $converged \n")

    return pz, pd, converged

end


"""
"""
function split_opt_params_and_close_n(pz::Dict{}, pd::Dict{}, data::Dict{}; n::Int=53, state::String="state")

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])
    p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], pd[state]), fit_vec)

    parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec))
    ll(x) = ll_wrapper_n(x, data, parameter_map_f, n=n)

    return p_opt, ll, parameter_map_f

end


"""
"""
function ll_wrapper_n(p_opt::Vector{TT}, data::Dict, parameter_map_f::Function;
        n::Int=53) where {TT <: Any}

    pz, pd = parameter_map_f(p_opt)
    -compute_LL_n(pz, pd, data; n=n)

end


"""
"""
compute_LL_n(pz::Vector{T}, pd::Vector{T}, data; n::Int=53) where {T <: Any} = sum(LL_all_trials_n(pz, pd, data, n=n))


"""
"""
function LL_all_trials_n(pz::Vector{TT}, pd::Vector{TT}, data::Dict; n::Int=53) where {TT <: Any}

    bias, lapse = pd
    σ2_i, B, λ, σ2_a, σ2_s, ϕ, τ_ϕ = pz
    L, R, nT, nL, nR, choice = [data[key] for key in ["left","right","nT","binned_left","binned_right","pokedR"]]
    dt = data["dt"]

    P,M,xc,dx = initialize_latent_model(σ2_i, B, λ, σ2_a, n, dt, L_lapse=lapse/2, R_lapse=lapse/2)

    pmap((L,R,nT,nL,nR,choice) -> LL_single_trial!(λ, σ2_a, σ2_s, ϕ, τ_ϕ,
        P, M, dx, xc, L, R, nT, nL, nR, choice, bias, n, dt), L, R, nT, nL, nR, choice)

end


"""
"""
function initialize_latent_model(σ2_i::TT, B::TT, λ::TT, σ2_a::TT,
     n::Int, dt::Float64; L_lapse::TT=0., R_lapse::TT=0.) where {TT <: Any}

    #bin centers and number of bins
    xc,dx = bins(B,n)

    # make initial latent distribution
    P = P0(σ2_i,n,dx,xc,dt; L_lapse=L_lapse, R_lapse=R_lapse)

    # build state transition matrix for times when there are no click inputs
    M = transition_M(σ2_a*dt,λ,zero(TT),dx,xc,n,dt)

    return P, M, xc, dx

end


"""
"""
function bins(B::TT,n::Int) where {TT}

    dx = 2. *B/(n-2);  #bin width

    xc = vcat(collect(range(-(B+dx/2.),stop=-dx,length=Int((n-1)/2.))),0.,
        collect(range(dx,stop=(B+dx/2.),length=Int((n-1)/2)))); #centers

    return xc, dx

end


"""
    compute_CIs_n!(pz, pd, data)

Computes confidence intervals based on the likelihood ratio test
"""
function compute_CIs_n!(pz::Dict, pd::Dict, data::Dict; state::String="final")

    ll_θ = compute_LL_n(pz[state], pd[state], data) 

    lb = combine_latent_and_observation(pz["lb"], pd["lb"])
    ub = combine_latent_and_observation(pz["ub"], pd["ub"])

    fit_vec = combine_latent_and_observation(pz["fit"], pd["fit"])

    CI = Vector{Vector{Float64}}(undef,length(fit_vec))

    for i = 1:length(fit_vec)
        
        println(i)

        fit_vec2 = falses(length(fit_vec))
        fit_vec2[i] = true
            
        p_opt, p_const = split_variable_and_const(combine_latent_and_observation(pz[state], 
                pd[state]), fit_vec2)

        parameter_map_f(x) = split_latent_and_observation(combine_variable_and_const(x, p_const, fit_vec2))
        ll(x) = -ll_wrapper_n([x], data, parameter_map_f) - (ll_θ - 1.92)
        
        xs = range(lb[i], stop=ub[i], length=50)
        samps = map(x->ll(x), xs)
        idxs = findall(diff(sign.(samps)) .!= 0)

        #CI[i] = sort(find_zeros(ll, lb[i], ub[i]; naive=true, no_pts=3))

        CI[i] = []
        for j = 1:length(idxs)
            @time newroot = find_zero(ll, (xs[idxs[j]], xs[idxs[j]+1]), Bisection())
            push!(CI[i], newroot)
        end

        if length(CI[i]) > 2
            @warn "More than three roots found. Uh oh."
        end

        if length(CI[i]) == 0
            CI[i] = vcat(lb[i], ub[i])
        end

        if length(CI[i]) == 1
            if CI[i][1] < p_opt[1]
                CI[i] = sort(vcat(CI[i], ub[i]))
            elseif CI[i][1] > p_opt[1]
                CI[i] = sort(vcat(CI[i], lb[i]))
            end
        end

    end

    pz["CI_plus"], pd["CI_plus"] = split_latent_and_observation(map(ci-> ci[2], CI))
    pz["CI_minus"], pd["CI_minus"] = split_latent_and_observation(map(ci-> ci[1], CI))

    return pz, pd, CI

end

function compute_Hessian_n(pz::Dict{}, pd::Dict{}, data::Dict{};
    n::Int=53, state::String="state") where {TT <: Any}

    println("computing Hessian! \n")
    p_opt, ll, = split_opt_params_and_close_n(pz,pd,data; n=n,state=state)
    ForwardDiff.hessian(ll, p_opt)

end


"""
    compute_CIs_n!(pz, pd, H)
"""
function compute_CIs_n!(pz::Dict, pd::Dict, H::Array{Float64,2})

    println("computing confidence intervals \n")

    gooddims = 1:size(H,1)

    evs = findall(eigvals(H[gooddims,gooddims]) .<= 0)
    otherbad = vcat(map(i-> findall(abs.(eigvecs(H[gooddims,gooddims])[:,evs[i]]) .> 0.5), 1:length(evs))...)
    gooddims = setdiff(gooddims,otherbad)

    p_opt, ll, parameter_map_f = split_opt_params_and_close_n(pz,pd,Dict(); state="final")

    CI = fill!(Vector{Float64}(undef,size(H,1)),1e8);

    CI[gooddims] = 2*sqrt.(diag(inv(H[gooddims,gooddims])))

    pz["CI_plus"], pd["CI_plus"] = parameter_map_f(p_opt + CI)
    pz["CI_minus"], pd["CI_minus"] = parameter_map_f(p_opt - CI)

    return pz, pd

end