ll_wrapper_RBF_1cell(w,rbf,c,SC,nT,dt) = -sum(poiss_LL.(vcat(SC...), 
        log.(1. .+ exp.(vcat(map(x->rbf(c[1:x]) * w, nT)...))),dt))

function do_optim_RBF_1cell(w0::Vector{Float64},dt::Float64,nT::Vector{Int},SC::Vector{Vector{Int}},rbf,c)
            
        ll(w) = ll_wrapper_RBF_1cell(w,rbf,c,SC,nT,dt)
        opt_output, state = opt_ll(w0,ll;g_tol=1e-6,x_tol=1e-12,f_tol=1e-12,
            iterations=1000,show_trace=false)
        w = Optim.minimizer(opt_output)
                            
end