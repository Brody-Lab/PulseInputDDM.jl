using pulse_input_DDM

println("using ", nprocs(), " cores")
data_path, save_path = "../data/dmFC_muscimol/", "../results/dmFC_muscimol/"
isdir(save_path) ? nothing : mkpath(save_path)

files = readdir(data_path)
files = files[.!isdir.(files)]
file = files[1]

data = load_choice_data(data_path, file)

pz, pd = default_parameters()

if isfile(save_path*file)
    pz, pd = reload_optimization_parameters(save_path, file, pz, pd)    
end

pz, pd, = optimize_model(pz, pd, data)
H = compute_Hessian(pz, pd, data; state="final")
pz, pd = compute_CIs!(pz, pd, H)

save_optimization_parameters(save_path,file,pz,pd)
