
using MAT, module_DDM, LineSearches, StatsBase
using ForwardDiff: gradient!, hessian, gradient
using Optim

ratname = ARGS[1]; #which rat
sessid = ARGS[2]; #which sessid
in_pth = ARGS[3]; #location of data
out_pth = ARGS[4]; #where to save results
model_type = ARGS[5]; #which model to fit

use = Dict("choice" => false, "spikes" => false)

if model_type == "choice"
    use["choice"] = true
elseif model_type == "spikes"
    use["spikes"] = true
elseif model_type == "joint"
    use["choice"] = true
    use["spikes"] = true
end

file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
data = read(file, "data")
close(file)

#converts binned data into useable form for julia
convert_data!(data)

file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
x0 = vec(read(file, "x0"))
lower = vec(read(file,"lb"))
upper = vec(read(file,"ub"))
xi = vec(read(file,"xf"))
fit_vec = convert(Array{Bool},vec(read(file, "fit_vec")))
close(file)

if isfile(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat")
    file = matopen(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat")
    history = read(file,"history")
    xi = vec(history[:,end])
    close(file)
end

start_time = time()

function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64}(sum(fit_vec),0)
    for i = 1:length(os)
        history= cat(2,history,os[i].metadata["x"])
    end
    matwrite(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("history" => history))

    if so_far > 60 * 60 * 23.5
        return true
    else
        return false
    end

end

od = OnceDifferentiable(x -> LL_all_trials_unc(x,data,use,x0,fit_vec,lower,upper), xi; autodiff = :forward)

@time results = optimize(od, xi, BFGS(), Optim.Options(time_limit = 3600 * 23.5, g_tol = 1e-32, x_tol = 1e-32, f_tol = 1e-32, iterations = 1000, store_trace = true, show_trace = true, extended_trace = true, callback = my_callback, allow_f_increases = true))

println(results)
xf = results.minimizer;
history = Optim.x_trace(results)

H = hessian(x -> LL_all_trials_unc(x, data, use, x0, fit_vec, lower, upper), xf)

# save results
matwrite(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("H" => H, "history" => history, "xf" => xf))

