
using MAT, module_DDM_v3, LineSearches
using ForwardDiff: gradient!, hessian, gradient
using Optim

ratname = ARGS[1]; #which rat
sessid = ARGS[2]; #which sessid
in_pth = ARGS[3]; #location of data
out_pth = ARGS[4]; #where to save results
model_type = ARGS[5]; #which model to fit
reload = ARGS[6]; #which xf to reload from

file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
data = read(file, "data")
close(file)

#converts binned data into useable form for julia
convert_data!(data,model_type)

#load static/initial x values and fit_vec (which parameters to fit)
file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
x0 = vec(read(file, "x0"))
fit_vec = convert(Array{Bool},vec(read(file, "fit_vec")))
close(file)

if reload == "matlab"
    file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
    try
        xf = vec(read(file,"xf"))
    catch
        history = read(file,"history")
        xf = vec(history["x"][:,end])
    end
    close(file)
elseif reload == "julia"
    try
        file = matopen(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat")
        xf = read(file,"xf")
    catch
        file = matopen(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat")
        history = read(file,"history")
        xf = vec(history[:,end])
    end
    close(file)
end

start_time = time()

function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64}(sum(fit_vec),0)
    for i = 1:length(os)
        history= cat(2,history,reparameterize(os[i].metadata["x"],x0,fit_vec,model_type))
    end
    matwrite(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("history" => history))

    if so_far > 60 * 60 * 23.0
        return true
    else
        return false
    end

end

x0 = deparameterize(xf,x0,fit_vec,model_type)
xf = x0[fit_vec]

od = OnceDifferentiable(x -> ll_wrapper(x,data,model_type,x0,fit_vec), xf; autodiff = :forward)

@time results = optimize(od, xf, BFGS(), Optim.Options(time_limit = 3600 * 23., g_tol = 1e-4, x_tol = 1e-32, f_tol = 1e-32, iterations = 1000, store_trace = true, show_trace = true, extended_trace = true, callback = my_callback, allow_f_increases = true))

println(results)
xf = results.minimizer;
history = Optim.x_trace(results)

H = hessian(x -> ll_wrapper(x, data, model_type, x0, fit_vec), xf)

# save results
matwrite(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("H" => H, "history" => history, "xf" => xf))

