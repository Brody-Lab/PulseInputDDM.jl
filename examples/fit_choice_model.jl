#use the resources of the package
using pulse_input_DDM

println("using ", nprocs(), " cores")

#define useful paths to the data and to a directory to save results
data_path, save_path = "../data/dmFC_muscimol/", "../results/dmFC_muscimol/"

#if the directory that you are saving to doesn't exist, make it
isdir(save_path) ? nothing : mkpath(save_path)

#read the name of the file located in the data directory
files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

#load your data
data = load_choice_data(data_path, file)

#generate default parameters for initializing the optimization
pz, pd = default_parameters()

#if you've already ran the optimization once and want to restart from where you stoped, this will reload those parameters
if isfile(save_path*file)
    pz, pd = reload_optimization_parameters(save_path, file, pz, pd)
end

#run the optimization
pz, pd, = optimize_model(pz, pd, data)

#compute the Hessian around the ML solution, for confidence intervals
H = compute_Hessian(pz, pd, data; state="final")

#compute confidence intervals
pz, pd = compute_CIs!(pz, pd, H)

#save results
save_optimization_parameters(save_path,file,pz,pd)
