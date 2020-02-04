"""
    optimize(x, ll;
        g_tol=1e-3, x_tol=1e-10, f_tol=1e-6,
        iterations=Int(5e3),
        show_trace=true, extended_trace=false)

Wrapper for executing an unconstrained optimization based on the objective function ll. x is the initial starting point.
"""
function optimize(x::Vector{TT}, ll, lb, ub;
        g_tol::Float64=1e-12, x_tol::Float64=1e-16, f_tol::Float64=1e-16,
        iterations::Int=Int(5e3), outer_iterations::Int=Int(1e1), 
        show_trace::Bool=true, extended_trace::Bool=false,
        show_every::Int=2, scaled::Bool=false) where TT <: Real

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=scaled), linesearch = BackTracking())
    
    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol,
        outer_iterations= outer_iterations, allow_outer_f_increases=true,
        show_every=show_every)

    output = Optim.optimize(obj, lb, ub, x, Fminbox(m), options)

    return output

end


"""
    stack(x,c)

Combine two vector into one. The first vector is variables for optimization, the second are constants.
"""
function stack(x::Vector{TT}, c::Vector{Float64}, fit::Union{BitArray{1},Vector{Bool}}) where TT

    v = Vector{TT}(undef,length(fit))
    v[fit] = x
    v[.!fit] = c

    return v

end


"""
    unstack(v)

Break one vector into two. The first vector is variables for optimization, the second are constants.
"""
function unstack(v::Vector{TT}, fit::Union{BitArray{1},Vector{Bool}}) where TT

    x,c = v[fit], v[.!fit]

end
