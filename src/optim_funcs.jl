"""
    check_pz(pz)

    Checks the values of pz, looking for very specific parameters where singularities occur in the model.

"""
function check_pz(pz)

    if (pz["state"][pz["name"] .== "ϕ"][1] == 1.) & pz["fit"][pz["name"] .== "ϕ"][1]
        error("ϕ has a value of 1. and you are optimizing w.r.t. to it
            but this code ignores ϕ when it is exactly 1.
            Change your initialization of ϕ.")
    end

    if (pz["state"][pz["name"] .== "λ"][1] == 1.) & pz["fit"][pz["name"] .== "λ"][1]
        error("λ has a value of 0. and you are optimizing w.r.t. to it
            but this code ignores λ when it is exactly 0.
            Change your initialization of λ.")
    end

end

split_variable_and_const(p::Vector{TT}, fit_vec::Union{BitArray{1},Vector{Bool}}) where TT = p[fit_vec],p[.!fit_vec]

function combine_variable_and_const(p_opt::Vector{TT}, p_const::Vector{Float64}, 
            fit_vec::Union{BitArray{1},Vector{Bool}}) where TT
    
    p = Vector{TT}(undef,length(fit_vec))
    p[fit_vec] = p_opt;
    p[.!fit_vec] = p_const;
    
    return p
    
end

function opt_func_fminbox(x, ll, lb, ub;
        g_tol::Float64=1e-12, x_tol::Float64=1e-16, f_tol::Float64=1e-16,
        iterations::Int=Int(5e3),
        show_trace::Bool=true, extended_trace::Bool=false)

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())

    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol,
        outer_iterations= iterations, allow_outer_f_increases=true)

    output = Optim.optimize(obj, lb, ub, x, Fminbox(m), options)

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
