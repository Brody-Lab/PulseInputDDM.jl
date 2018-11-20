
using MAT, module_DDM_v2
using ForwardDiff: gradient!, hessian, gradient, Dual, partials, value

ratname="T036"
sessid="157201_157357_157507_168499"
in_pth="~/Dropbox/hanks_data_session"
model_type="spikes"
out_pth="/Users/briandepasquale/Dropbox/results/new_settling/"*sessid

use = Dict("choice" => false, "spikes" => false)
use["spikes"] = true

file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
data = read(file, "data")
close(file)

#converts binned data into useable form for julia
convert_data!(data)

file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
x0 = vec(read(file, "x0"))
#try
xf = vec(read(file,"xf"))
#catch
#    history = read(file,"history")
#    xf = vec(history["x"][:,end])
#end

phi = Dual(1.0,1.0)

for i = 1:length(data["T"])
    La, Ra = make_adapted_clicks(data["leftbups"][i],data["rightbups"][i],phi,x0[8])
        for j = 1:length(La)
            if isnan(partials(La[j])[1])
                println(i)
            end
        end

        for j = 1:length(Ra)
            if isnan(partials(Ra[j])[1])
                println(i)
            end
        end
end

fit_vec = convert(Array{Bool},vec(read(file, "fit_vec")))
LL = LL_all_trials(xf,data,use,x0,fit_vec)
println(LL)
g = gradient(x -> LL_all_trials(x, data, use, x0, fit_vec), xf)
H = hessian(x -> LL_all_trials(x,data,use,x0,fit_vec),xf)

# save results
#save(out_pth*"/H_"*ratname*"_"*model_type*".jld", "H", H, "LL", LL);
matwrite(out_pth*"/gML_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("g" => g, "LL" => LL))
