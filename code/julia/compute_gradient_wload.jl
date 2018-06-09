
using MAT, module_DDM_v3
using ForwardDiff: gradient!, hessian, gradient, GradientConfig, Chunk

ratname="T036"
sessid="157201_157357_157507_168499"
in_pth="/Users/briandepasquale/Projects/inProgress/spike-data_latent-accum/data/hanks_data_sessions"
model_type="spikes"
out_pth="/Users/briandepasquale/Projects/inProgress/spike-data_latent-accum/data/results/julia/"*model_type*"/"*sessid*"/17813002"

#ratname = ARGS[1]; #which rat
#sessid = ARGS[2]; #which sessid
#in_pth = ARGS[3]; #location of data
#out_pth = ARGS[4]; #where to save results
#model_type = ARGS[5]; #which model to fit

file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
data = read(file, "data")
close(file)
#converts binned data into useable form for julia
convert_data!(data,model_type)
N = data["Ntotal"]
fit_vec = fit_func(model_type,N)

file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
x0 = vec(read(file, "x0"))
close(file)

#   try
#       file = matopen(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat")
#       xf = vec(read(file,"xf"))
#    catch
#        file = matopen(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat")
#        history = read(file,"history")
#        xf = vec(history[:,end])
#    end
#close(file)

@everywhere LL(x) = ll_wrapper(x,data,model_type,x0,fit_vec,N)
xf = x0[fit_vec]
x0 = deparameterize(xf,x0,fit_vec,model_type,N)
xf = x0[fit_vec]
myLL = LL(xf)
println(myLL)

cfg = GradientConfig(LL, xf, Chunk{length(xf)}())
@time g = gradient(LL, xf, cfg)

#function blah(lambda)

#    mu = exp(lambda*2e-2)*1. + 1./lambda * (exp(lambda*2e-2) - 1.)

#    return mu

#end

# save results
#matwrite(out_pth*"/gML_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("g" => g, "LL" => LL))
