
using MAT, module_DDM_v4, LineSearches, Optim, JLD, HDF5
using ForwardDiff: gradient!, hessian, gradient, GradientConfig, Chunk

ratname = ARGS[1]; #which rat
model_type = ARGS[2]; #which model to fit
path = ARGS[3]; #parent directory where data and results are located
out_pth = ARGS[4]; #directory where data will be saved
reload_pth = ARGS[5]; #directory to reload history from

in_pth = path*"/data/hanks_data_sessions"

if model_type == "joint"
    model_type = ["spikes","choice"]
end

if ratname == "B053"
    sessid = [51344, 52162,52425, 53154,54377]    
elseif ratname == "B068"
    sessid = [46331,46484,46630,47745,48117,48988,49313,49380,49545,49819,50353,50838,50951,51621,51852,52307]
elseif ratname == "T011"
    sessid = [153219,153382,153510,154806,154950,155375,155816,155954,157026,157178,157483,158231,161057,161351,164574,165972]   
elseif ratname == "T034"
    sessid = [151801,152157,152272,152370,152509,152967,153253,153859,154807,154949,
        157482,161543,163862,163993,164420,164573,165023,168608,169683];
elseif ratname == "T035"
    #sessid = [152177,153274,153536,154288,154440,155839,156150,156443,156896,
    #    157200,157359,161394,162142, 162258,163098,163885,164449,164604,164752,164900,165058,166135,
    #    166590,167725,167855,167993,168132,168628,169448,169736,169873,169993]
    sessid = [169448,167725,166135,164900]
elseif ratname == "T036"
    #sessid = [154154,154291,154448,154991,155124,155247,155840,157201,157357,157507,168499,168627]
    sessid = [157201,157357,157507,168499]
elseif ratname == "T063"
    sessid = [191956,193936,194554,194649,194770,194898,195271,195546,195676,195791,196336,196580,
        196708,197075,197212,197479,198004]
elseif ratname == "T068"
    sessid = [195545,195790,196335,196579,196709,197204,197478,198003,198137,198249]
end

if isfile(reload_pth*"/results.jld")
    
    #open for re-writing
    results_file = jldopen(reload_pth*"/results.jld", "r")

    #load data
    data = read(results_file,"data")
    N = read(results_file,"N")
    x0 = read(results_file,"x0")
    fit_vec = read(results_file,"fit_vec") 
    p_opt = read(results_file,"p_opt")

    #close reload file
    close(results_file)

    #open for writing
    results_file = jldopen(out_pth*"/results.jld", "w")
    #write sessid to
    write(results_file, "sessid", sessid)
    close(results_file)
    #re-open for rewriting
    results_file = jldopen(out_pth*"/results.jld", "r+")

    #save data and number of neurons
    write(results_file, "data", data)
    write(results_file, "N", N)
    write(results_file, "x0", x0)
    write(results_file, "fit_vec", fit_vec)
    
else
    
    #open for writing
    results_file = jldopen(out_pth*"/results.jld", "w")
    #write sessid to
    write(results_file, "sessid", sessid)
    close(results_file)
    #re-open for rewriting
    results_file = jldopen(out_pth*"/results.jld", "r+")
        
    #create new data
    data = Dict("leftbups" => Array{Float64}[], "rightbups" => Array{Float64}[], 
                "T" => Float64[], "hereL" => Array{Int64}[], "hereR" => Array{Int64}[],
                "nT" => Int64[], "pokedR" => Bool[], "spike_counts" => Array{Int64}[], 
                "N" => Array{Int64}[], "correct_dir" => Bool[])

    t0 = 0;
    N0 = 0;

    for i = 1:length(sessid)
        file = matopen(in_pth*"/"*ratname*"_"*string(sessid[i])*".mat")
        rawdata = read(file,"rawdata")
        close(file)
        data,t0,N0 = package_data!(data,rawdata,model_type,t0,N0)
    end
    N = N0
    
    x0 = compute_x0(data,model_type,N)
    fit_vec = fit_func(model_type,N) 
    p_opt = vec(x0[fit_vec])

    #save data and number of neurons
    write(results_file, "data", data)
    write(results_file, "N", N)
    write(results_file, "x0", x0)
    write(results_file, "fit_vec", fit_vec)

end
    
p_const = x0[.!fit_vec]

#convert p_const and p_opt to unbounded domain
p = group_params(p_opt, p_const, fit_vec)
p = bounded_to_inf!(p,model_type,N=N)
p_opt, p_const = break_params(p, fit_vec)

start_time = time()

function my_callback(os)

    so_far = time() - start_time
    println(" * Time so far:     ", so_far)
  
    history = Array{Float64,2}(sum(fit_vec),0)
    history_gx = Array{Float64,2}(sum(fit_vec),0)
    for i = 1:length(os)
        ptemp = group_params(os[i].metadata["x"], p_const, fit_vec)
        ptemp = inf_to_bounded!(ptemp,model_type,N=N)
        ptemp_opt, = break_params(ptemp, fit_vec)       
        history = cat(2,history,ptemp_opt)
        history_gx = cat(2,history_gx,os[i].metadata["g(x)"])
    end
    save(out_pth*"/history.jld", "history", history, "history_gx", history_gx)

    return false

end

@everywhere LL(x) = ll_wrapper(x, p_const, fit_vec, data, model_type, N=N, beta=Dict("d"=>1e-6))
 
od = OnceDifferentiable(LL,p_opt; autodiff=:forward)

@time results = optimize(od, p_opt, BFGS(linesearch = BackTracking()), Optim.Options(time_limit = 3600 * 46., g_tol = 1e-12, x_tol = 1e-16, f_tol = 1e-16, 
                                                                                 iterations = 100000, store_trace = true, show_trace = true, 
                                                                                extended_trace = true,  callback = my_callback, allow_f_increases = true))

println(results)
p_opt = results.minimizer; #final result in the unbounded domain

p = group_params(p_opt, p_const, fit_vec)
p = inf_to_bounded!(p,model_type,N=N) #convert p_opt to bounded domain
p_opt, = break_params(p, fit_vec)

#save ML parameters
write(results_file, "p_opt", p_opt)
#close file 
close(results_file)
