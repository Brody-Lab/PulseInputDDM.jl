
using MAT, module_DDM_v3, JLD
using ForwardDiff: gradient!, hessian, gradient

ratname = ARGS[1]; #which rat
sessid = ARGS[2]; #which sessid
in_pth = ARGS[3]; #location of data
out_pth = ARGS[4]; #where to save results
model_type = ARGS[5]; #which model to fit

file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
data = read(file, "data")
close(file)

#converts binned data into useable form for julia
convert_data!(data,use)

file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
x0 = vec(read(file, "x0"))
#try
xf = vec(read(file,"xf"))
#catch
#    history = read(file,"history")
#    xf = vec(history["x"][:,end])
#end

fit_vec = convert(Array{Bool},vec(read(file, "fit_vec")))
LL = LL_all_trials(xf,data,use,x0,fit_vec)
println(LL)
H = hessian(x -> LL_all_trials(x, data, use, x0, fit_vec), xf)

# save results
matwrite(out_pth*"/H_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("H" => H, "LL" => LL))
