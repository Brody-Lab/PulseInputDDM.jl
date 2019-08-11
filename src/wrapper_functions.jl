

#################################### Optimization #################################

function check_pz!(pz)
    
    if (pz["state"][6] == 1.) & pz["fit"][6]
        error("ϕ has a value of 1. and you are optimizing w.r.t. to it 
            but this code ignores ϕ when it is exactly 1. 
            Change you initialization of ϕ.")
    end
    
    if (pz["state"][3] == 1.) & pz["fit"][3]
        error("λ has a value of 0. and you are optimizing w.r.t. to it
            but this code ignores λ when it is exactly 0. 
            Change you initialization of λ.")
    end
    
    if any(pz["state"] .== pz["lb"])
        @warn "some parameter(s) at lower bound. bumped it (them) up 1/4 from the lower bound."
        pz["state"][pz["state"] .== pz["lb"]] .= 
            pz["lb"][pz["state"] .== pz["lb"]] .+ 
            0.25 .* (pz["ub"][pz["state"] .== pz["lb"]] .- pz["lb"][pz["state"] .== pz["lb"]])
    end
    
    if any(pz["state"] .== pz["ub"])
        @warn "some parameter(s) at upper bound. bumped it (them) down 1/4 from the upper bound."
        pz["state"][pz["state"] .== pz["ub"]] .= 
            pz["ub"][pz["state"] .== pz["ub"]] .- 
            0.25 .* (pz["ub"][pz["state"] .== pz["ub"]] .- pz["lb"][pz["state"] .== pz["ub"]])
    end
    
    return pz
    
end

function opt_ll_con(p_opt, ll, lb, ub;
        g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16,iterations::Int=Int(5e3),
        show_trace::Bool=true, extended_trace::Bool=false,
        outer_iterations::Int=Int(5e3))
    
    obj = OnceDifferentiable(ll, p_opt; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())
    
    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol, 
        iterations= iterations, allow_f_increases=true, 
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol, 
        outer_iterations= outer_iterations, allow_outer_f_increases=true)
    
    #lbfgsstate = Optim.initial_state(m, options, obj, lb, ub, p_opt)
    
    output = Optim.optimize(obj, lb, ub, p_opt, Fminbox(m), options)
    
    return output
    
end



function opt_ll(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16,iterations::Int=Int(5e3),
        show_trace::Bool=true, extended_trace::Bool=false)
    
    obj = OnceDifferentiable(ll, p_opt; autodiff=:forward)
    #obj = TwiceDifferentiable(ll, p_opt; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())
    #m = GradientDescent(alphaguess = InitialStatic(alpha=1e-2,scaled=true), linesearch = Static())
    #m = BFGS(alphaguess = InitialHagerZhang(α0=1.0), linesearch = HagerZhang())
    #m = Newton(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
    #    linesearch = BackTracking())
    #m = Newton(alphaguess = InitialHagerZhang(α0=1.0), linesearch = HagerZhang())
    
    options = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = iterations, store_trace = true, show_trace = show_trace, 
        extended_trace = extended_trace, allow_f_increases = true)
    lbfgsstate = Optim.initial_state(m, options, obj, p_opt)
    
    output = Optim.optimize(obj, p_opt, m, options, lbfgsstate)
    
    return output, lbfgsstate
    
end

function opt_ll_Newton(p_opt,ll;g_tol::Float64=1e-12,x_tol::Float64=1e-16,f_tol::Float64=1e-16,iterations::Int=Int(5e3),
        show_trace::Bool=true)
    
    obj = OnceDifferentiable(ll, p_opt; autodiff=:forward)
    #m = Newton(alphaguess = LineSearches.InitialStatic(alpha=1.0,scaled=true), 
    #    linesearch = BackTracking())
    m = Newton(alphaguess = InitialHagerZhang(α0=1.0), linesearch = HagerZhang())
    options = Optim.Options(g_tol = g_tol, x_tol = x_tol, f_tol = f_tol, 
        iterations = iterations, store_trace = true, show_trace = show_trace, 
        extended_trace = false, allow_f_increases = true)
    lbfgsstate = Optim.initial_state(m, options, obj, p_opt)

    output = Optim.optimize(obj, p_opt, m, options, lbfgsstate)

    return output, lbfgsstate
    
end

########################## Priors ###########################################

gauss_prior(p::Vector{TT}, mu::Vector{Float64}, beta::Vector{Float64}) where {TT} = -sum(beta .* (p - mu).^2)

#=

function splitdata_train_test(data, trunc_μ_λ; frac::Float64=0.8, dt::Float64=1e-2)
    
    divided_data, divided_λ = train_test_divide(data, trunc_μ_λ, frac)
    
    pz, py = load_and_optimize(divided_data["train"], divided_data["train"]["N0"]; 
        fit_latent= vcat(trues(7)), fit_initial=vcat(1e-1,15.,-0.1,10.,1.,0.2,0.005), 
        show_trace= true, λ0=divided_λ["train"], f_str="softplus",dimy=3)
    
    LL_stim = compute_LL(pz[:final], py[:final], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    #LL_null = compute_LL(pz[:final], [vcat(py[:final][1][1],0.,py[:final][1][3])], 
    #        divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    LL_null = neural_null(vcat(vcat(divided_data["test"]["spike_counts"]...)...),
        vcat(vcat(divided_λ["test"]["by_trial"]...)...), dt) 

    LL_stim - LL_null
    
end

function splitdata_train_test_multi(data, trunc_μ_λ; frac::Float64=0.9)
    
    divided_data, divided_λ = train_test_divide_multi(data ,trunc_μ_λ, frac)
    
    pz, py = load_and_optimize(divided_data["train"], divided_data["train"]["N0"]; 
        fit_latent= vcat(falses(1),trues(4),falses(2)), 
        show_trace= false, λ0=divided_λ["train"], f_str="softplus", dimy = 3)
    
    LL_stim = compute_LL(pz[:final], py[:final], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")
    
    LL_null = compute_LL(pz[:final], [vcat(py[:final][1][1],0.,py[:final][1][3])], 
            divided_data["test"]; λ0=divided_λ["test"]["by_trial"], f_str="softplus")

    LL_stim - LL_null
    
end

=#
