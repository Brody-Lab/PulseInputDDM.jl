using pulse_input_DDM

println("using ", nprocs(), " cores")
data_path, save_path = "../data/dmFC_muscimol/", "../results/dmFC_muscimol_8p/"
isdir(save_path) ? nothing : mkpath(save_path)

files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

data = load_choice_data(data_path, file)

if isfile(save_path*file)
    pz, pd = reload_optimization_parameters(save_path, file, pz, pd)    
end

pz, pd, = optimize_model(pz, pd, data)
H = compute_Hessian(pz, pd, data)
pz, pd = comute_CIs!(pz, dt, H)

save_optimization_parameters(save_path,file,pz,pd)

