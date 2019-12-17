#use the resources of the package
using pulse_input_DDM

println("using ", nprocs(), " cores")

#define useful paths to the data and to a directory to save results
data_file = "../data/dmFC_muscimol/file.mat"
tail = ""

#load your data
data = load_choice_data(data_file)

save_path = "../results/dmFC_muscimol/"

#if the directory that you are saving to doesn't exist, make it
if !isdir(save_path)
    mkpath(save_path)
end

#if you've already ran the optimization once and want to restart from where you stoped, this will reload those parameters
if isfile(save_path*tail)
    pz, pd = reload_optimization_parameters(save_path, tail, pz, pd)
end

#generate default parameters for initializing the optimization
options = opt(x0=vcat([0.1, 15., -0.1, 20., 0.5, 0.8, 0.008], [0.,0.01]))

#run the optimization
model, options = optimize(data; options=options)

#compute the Hessian around the ML solution, for confidence intervals
H = Hessian(model)

#compute confidence intervals
CI = CIs(model, H)

#save results
#save_optimization_parameters(save_path,tail,pz,pd)
