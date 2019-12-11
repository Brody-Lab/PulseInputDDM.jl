"""
    x_c(p_opt, p_const)

Combine two vector into one. The first vector is variables for optimization, the second are constants.
"""
function x_c(x::Vector{TT}, c::Vector{Float64}, fit::Union{BitArray{1},Vector{Bool}}) where TT

    p = Vector{TT}(undef,length(fit))
    p[fit] = x
    p[.!fit] = c

    return p

end


"""
    opt_func_fminbox(x, ll, lb, ub;
        g_tol=1e-3, x_tol=1e-10, f_tol=1e-6,
        iterations=Int(5e3), outer_iterations=Int(1e1)
        show_trace=true, extended_trace=false)

Wrapper for executing a constrained optimization based on the objective function ll. x is the initial starting point.
"""
function opt_func_fminbox(x, ll, lb, ub;
        g_tol::Float64=1e-12, x_tol::Float64=1e-16, f_tol::Float64=1e-16,
        iterations::Int=Int(5e3), outer_iterations::Int=Int(1e1),
        show_trace::Bool=true, extended_trace::Bool=false)

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())

    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol,
        outer_iterations= outer_iterations, allow_outer_f_increases=true)

    output = Optim.optimize(obj, lb, ub, x, Fminbox(m), options)

    return output

end


"""
    opt_func(x, ll;
        g_tol=1e-3, x_tol=1e-10, f_tol=1e-6,
        iterations=Int(5e3),
        show_trace=true, extended_trace=false)

Wrapper for executing an unconstrained optimization based on the objective function ll. x is the initial starting point.
"""
function opt_func(x, ll;
        g_tol::Float64=1e-12, x_tol::Float64=1e-16, f_tol::Float64=1e-16,
        iterations::Int=Int(5e3),
        show_trace::Bool=true, extended_trace::Bool=false)

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=true), linesearch = BackTracking())

    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        allow_outer_f_increases=true)

    output = Optim.optimize(obj, x, m, options)

    return output

end
