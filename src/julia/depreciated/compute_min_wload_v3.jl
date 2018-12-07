
using MAT, module_DDM_v3, LineSearches, Optim, JLD, HDF5
using ForwardDiff: gradient!, hessian, gradient, GradientConfig, Chunk

ratname = ARGS[1]; #which rat
sessid = ARGS[2]; #which sessid
model_type = ARGS[3]; #which model to fit
in_pth = ARGS[4]; #location of data
out_pth = ARGS[5]; #where to save results
reload_pth = ARGS[6]; #where to reload existing results from

#ratname="T036"
#sessid="157201_157357_157507_168499"
#in_pth="/Users/briandepasquale/Projects/inProgress/spike-data_latent-accum/data/hanks_data_sessions"
#model_type="spikes"
#reload_pth="/Users/briandepasquale/Projects/inProgress/spike-data_latent-accum/data/results/julia/"*model_type*"/"*sessid*"/17813002"
#out_pth="/Users/briandepasquale/Desktop"

data = Dict("leftbups" => Array{Float64}[], "rightbups" => Array{Float64}[], 
            "T" => Float64[], "hereL" => Array{Int64}[], "hereR" => Array{Int64}[],
            "nT" => Int64[], "pokedR" => Bool[], "spike_counts" => Array{Int64}[], 
            "N" => Array{Int64}[], "correct_dir" => Bool[])
sessidcell = split(sessid,"_")

t0 = 0;
N0 = 0;

for i = 1:length(sessidcell)
    file = matopen(in_pth*"/"*ratname*"_"*sessidcell[i]*".mat")
    rawdata = read(file,"rawdata")
    close(file)
    data,t0,N0 = package_data!(data,rawdata,model_type,t0,N0)
end
N = N0

#load or reload data
#try
#    file = matopen(reload_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
#    println("reloaded data")
#catch
#    file = matopen(out_pth*"/data_"*ratname*"_"*sessid*"_"*model_type*".mat");
#end
#data = read(file, "data")
#close(file)

#converts binned data into useable form for julia
#convert_data!(data,model_type)
#N = data["Ntotal"]

#load static/initial x values and fit_vec (which parameters to fit)
#try
#    file = matopen(reload_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
#catch
#    file = matopen(out_pth*"/"*ratname*"_"*sessid*"_"*model_type*".mat")
#end
#x0 = vec(read(file, "x0")) #loaded from bounded domain
#close(file)

x0 = compute_x0(data,model_type,N)

fit_vec = fit_func(model_type,N)

#reload existing xf in bounded domain
#try
#    file = matopen(reload_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat")
#    xf = vec(read(file,"xf"))
#    xf = xf[fit_vec]
#    close(file)
#    println("using xf")
#catch
#    try
#        file = matopen(reload_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat")
#        history = read(file,"history")
#        xf = vec(history[:,end])
#        println("using history")
#        close(file)
#     catch
        xf = vec(x0[fit_vec])
        println("using x0")
#     end
#end

#xf = vec(x0[fit_vec])
#convert x0 and xf to unbounded domain
x0 = deparameterize(xf,x0,fit_vec,model_type,N)
xf = x0[fit_vec] #and then redefine xf in the unbounded domain

start_time = time()

function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64}(sum(fit_vec),0)
    history_gx = Array{Float64}(sum(fit_vec),0)
    for i = 1:length(os)
        temp = reparameterize(os[i].metadata["x"],x0,fit_vec,model_type,N)
        #converts back to bounded domain
        temp = temp[fit_vec]
        history= cat(2,history,temp)
        history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    end
    matwrite(out_pth*"/julia_history_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("history" => history, "history_gx" => history_gx))

    if so_far > 60 * 60 * 23.0
        return true
    else
        return false
    end

end

#od = OnceDifferentiable(x -> ll_wrapper(x,data,model_type,x0,fit_vec,N), xf; autodiff = :forward)

@everywhere LL(x) = ll_wrapper(x,data,model_type,x0,fit_vec,N)

od = OnceDifferentiable(LL,xf; autodiff=:forward)

println(LL(xf))
#cfg = GradientConfig(LL, xf, Chunk{length(xf)}())
#@everywhere g(x) = gradient(LL, x, cfg)
#println(g(xf))


#g = gradient(x -> ll_wrapper(x,data,model_type,x0,fit_vec,N),xf)

@time results = optimize(od, xf, BFGS(linesearch = BackTracking()), Optim.Options(time_limit = 3600 * 23., g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                                                                                 iterations = 2500, store_trace = true, show_trace = true, 
                                                                                extended_trace = true,  callback = my_callback, allow_f_increases = true))

println(results)
xf = results.minimizer; #final result in the unbounded domain
#history = Optim.x_trace(results)

H = hessian(LL, xf)

xf = reparameterize(xf,x0,fit_vec,model_type,N) #convert xf to bounded domain
xf = xf[fit_vec] #define to contain only those parameters that were optimized
#x0 does not need to be converted, as it was saved in the bounded domain

x0 = reparameterize(x0[fit_vec],x0,fit_vec,model_type,N)

# save results
#matwrite(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".mat", Dict("H" => H, "xf" => xf)) 
save(out_pth*"/julia_"*ratname*"_"*sessid*"_"*model_type*".jld",
     "data", data, "x0", x0, "fit_vec", fit_vec, "xf", xf, "H", H, "N", N) 
