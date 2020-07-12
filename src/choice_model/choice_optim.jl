"""
    optimize_model(data, options, n; x_tol=1e-10, f_tol=1e-6, g_tol=1e-3,
        iterations=Int(2e3), show_trace=true)
Optimize model parameters. data is a type that contains the click data and the choices.
options is a type that contains the initial values, boundaries,
and specification of which parameters to fit.
BACK IN THE DAY TOLS WERE: x_tol::Float64=1e-4, f_tol::Float64=1e-9, g_tol::Float64=1e-2
"""
function optimize(data, data_dict, modeltype, options::DDMθoptions, dx::Float64;
        x_tol::Float64=1e-12, f_tol::Float64=1e-12, g_tol::Float64=1e-10,
        iterations::Int=Int(5e3), show_trace::Bool=true, outer_iterations::Int=Int(1e1),
        extended_trace::Bool=false, scaled::Bool=false)

    @unpack fit, lb, ub, x0 = options

    lb, = unstack(lb, fit)
    ub, = unstack(ub, fit)
    x0,c = unstack(x0, fit)

    ℓℓ(x) = objectivefn(modeltype, stack(x,c,fit), data, data_dict, dx)

    output = optimize(x0, ℓℓ, lb, ub; g_tol=g_tol, x_tol=x_tol,
        f_tol=f_tol, iterations=iterations, show_trace=show_trace,
        outer_iterations=outer_iterations, extended_trace=extended_trace,
        scaled=scaled)

    x = Optim.minimizer(output)
    x = stack(x,c,fit)
    θ = reconstruct_model(x, modeltype)
    model = choiceDDM(θ, data)
    ll = loglikelihood(θ, data, data_dict, dx)
    converged = Optim.converged(output)

    println("optimization complete. converged: $converged \n")

    return model, output, ll
end

"""
    objectivefn(modeltype, x::Vector{T1}, data, n::Int)
 Given a vector of parameters, data, n and modeltype returns negative log loglikelihood of the
 relevant modeltype    

"""
function objectivefn(modeltype, x::Vector{T1}, data, data_dict, dx::Float64) where {T1 <: Real}
    θ = reconstruct_model(x, modeltype)
    -loglikelihood(θ, data, data_dict,dx)
end
  
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
        scaled::Bool=false) where TT <: Real

    obj = OnceDifferentiable(ll, x; autodiff=:forward)
    m = BFGS(alphaguess = InitialStatic(alpha=1.0,scaled=scaled), linesearch = BackTracking())

    options = Optim.Options(g_tol=g_tol, x_tol=x_tol, f_tol=f_tol,
        iterations= iterations, allow_f_increases=true,
        store_trace = true, show_trace = show_trace, extended_trace=extended_trace,
        outer_g_tol=g_tol, outer_x_tol=x_tol, outer_f_tol=f_tol,
        outer_iterations= outer_iterations, allow_outer_f_increases=true)

    output = Optim.optimize(obj, lb, ub, x, Fminbox(m), options)

    return output

end
