"""
    optimize(x, ll, lb, ub)

Wrapper for executing an constrained optimization.

Arguments:

- `ll`: objective function. 
- `x`: an `array` of initial point
- `lb`: lower bounds. `array` the same length as `x`.
- `ub: upper bounds. `array` the same length as `x`.

"""
function optimize(x::Vector{TT}, ll, lb, ub;
        g_tol::Float64=1e-12, x_tol::Float64=1e-16, f_tol::Float64=1e-16,
        iterations::Int=Int(5e3), outer_iterations::Int=Int(1e1), 
        show_trace::Bool=true, extended_trace::Bool=false,
        scaled::Bool=false, time_limit::Float64=170000., 
        show_every::Int=10) where TT <: Real

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=scaled), linesearch = BackTracking())
    
    #start_time = time()
    #time_to_setup = zeros(1)
    #callback = x-> advanced_time_control(x, start_time, time_to_setup)

    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol,
        outer_iterations= outer_iterations, allow_outer_f_increases=true,
        time_limit = time_limit, show_every=show_every)

    output = Optim.optimize(obj, lb, ub, x, Fminbox(m), options)

    return output

end


"""
"""
function advanced_time_control(x, start_time, time_to_setup)
    println(" * Iteration:       ", x.iteration)
    so_far =  time()-start_time
    println(" * Time so far:     ", so_far)
    if x.iteration == 0
        time_to_setup[1] = time()-start_time
    else
        expected_next_time = so_far + (time()-start_time-time_to_setup[1])/(x.iteration)
        println(" * Next iteration ≈ ", expected_next_time)
        println()
        return expected_next_time < 60 ? false : true
    end
    println()
    false
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
