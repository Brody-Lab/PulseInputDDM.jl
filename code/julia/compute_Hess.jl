
using module_DDM_v4, JLD, HDF5
using ForwardDiff: hessian

model_type = ARGS[1]; #which model to fit
reload_pth = ARGS[2]; #directory to reload history from

#reload existing xf in bounded domain
p_opt,x0,N,fit_vec,data = load(reload_pth*"/results.jld","p_opt","x0","N","fit_vec","data")

p_const = x0[.!fit_vec]
#@everywhere LL(x) = ll_wrapper(x, p_const, fit_vec, data, model_type, N=N, beta=Dict("d"=>1e-6))
@everywhere LL(x) = LL_all_trials(x, p_const, fit_vec, data, model_type, N=N, beta=Dict("d"=>1e-6))

#p = group_params(p_opt, p_const, fit_vec)
#p = bounded_to_inf!(p,model_type,N=N)
#p_opt, p_const = break_params(p,fit_vec)

println(LL(p_opt))
H = hessian(LL, p_opt)

# save results
save(reload_pth*"/Hessian.jld","H", H) 
