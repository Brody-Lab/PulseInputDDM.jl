
using MAT, module_DDM, JLD, StatsBase
using ForwardDiff: gradient!, hessian, gradient

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
lb = vec(read(file, "lb"))
ub = vec(read(file, "ub"))
try
    xf = vec(read(file,"xf"))
catch
    history = read(file,"history")
    xf = vec(history["x"][:,end])
end

fit_vec = convert(Array{Bool},vec(read(file, "fit_vec")))
LL = LL_all_trials_unc(xf,data,use,x0,fit_vec,lb,ub)
println(LL)
H = hessian(x -> LL_all_trials_unc(x, data, use, x0, fit_vec, lb, ub), xf)

# save results
#save(out_pth*"/H_"*ratname*"_"*model_type*".jld", "H", H, "LL", LL);
matwrite(out_pth*"/H_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("H" => H, "LL" => LL))
