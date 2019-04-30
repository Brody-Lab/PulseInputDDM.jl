# a script for computing the model Hessian at the MLE point using autodiff
# does not actually perform optimization

# set up modules for analysis
using MAT
using ForwardDiff
include("compute_LL.jl")

# move to directory with the fits
cd("/home/alex/Dropbox/dynamic/model_fits/FITS")

# which rats to use, all with H0- prefix
##rats = [33 37 38 39 40 43 45 58 61 65 66 67 83 84];
##rats = ["B052", "B053", "B065", "B069", "B074", "B083", "B090", "B093", "B097", "B102", "B103", "B104", "B105", "B106", "B107", "B111", "B112", "B113", "B115"];
##rats = ["H034b", "H034d","H036b","H036d","H036a","H045b","H045d","H045a","H046b","H046d","H046a"];
##rats = ["H034d1","H034d2", "H036d1","H036d2","H036a1", "H036a2", "H045d1","H045d2","H045a1", "H045a2", "H046d1","H046d2","H046a1", "H046a2", "metaa1","metaa2","metaa", "metad1","metad2","metad", "metab"];
##rats = ["metad2_c1","metad2_c2","metad2_c3"];

#cd("/home/alex/Dropbox/dynamic/model_fits/OUTPUT/fit_analytical_no_prior_high_gamma")
#rats = ["H065","H067"];

rats = ["H037", "H066", "H084", "H129", "H140"];

# which model fitting group to look at
group = "ephys";

# Iterate over rats
for i = 1:length(rats) 
    # load model parameters at MLE point from matlab

    # for synthetic datasets
#    ratname = "dataset_H0"*string(rats[i]);
#    ratname = "dataset_"*string(i);

    # for bing rats
    ratname = rats[i];

    # for dynamic click rats
#    ratname = "H0"*string(rats[i]);
    
    try
    println("Now analyzing rat "*ratname)
    fit_data= matread(group*"/fit_analytical_"*ratname*".mat");
#    fit_data = matread(ratname*".mat");
    fit     = fit_data["fit"];
    data    = fit_data["data"];
    params  = fit["final"];
    
    # evaluate model LL just to make sure its correct
    NLL = compute_LL(data, params);
    if abs(NLL -  fit["f"]) > 1
        println("Oh no! Julia version is not within tolerance of MATLAB values")
        temp = NLL - fit["f"];
        println(temp)
    else
        println("Julia version with tolerance of MATLAB values")
    end

    # compute hessian using autodiff
    autodiff_hessian = ForwardDiff.hessian(x->compute_LL(data,x), params)

    # save new hessian
    matwrite(group*"/julia_hessian_$ratname.mat",Dict([("autodiff_hessian",autodiff_hessian)]))
   # matwrite("julia_hessian_$ratname.mat",Dict([("autodiff_hessian",autodiff_hessian)]))
    catch 
        println("Something went wrong")
    end
end



