# # Loading data and fitting a choice model
using pulse_input_DDM, Flatten, MAT, Random, JLD2
num_array = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
src_data_folder="/scratch/ejdennis/unfrozen_fclicks/";
fclicks_files=readdir(src_data_folder);
filename=fclicks_files[num_array]
data = load_choice_data(string(src_data_folder,filename));
n = 53
saved_file_location = "/scratch/ejdennis/ddm_runs/notfrozen/"
savedfilename = string(saved_file_location,"/",filename[1:end-4],num_array,"_results.mat")

function get_lls(rng,cross, fit, x0, xlb, xub, lb, ub, data,n,savefilename)
    # matread("/scratch/ejdennis/ddm_runs/notfrozen/K296_notfrozen_rawdata9_1_true_choicemodel_results.mat")
    # model = 
    ll = loglikelihood(model)
    io=open(string(savedfilename,"_",cross,"_ll_revised_",rng,".txt"),"w")
    println(io, string(ll))
    close(io)
end

output = map(rng -> get_lls(rng, false, fit, x0, data, n,savefilename), 1:20)
outputt = map(rng -> get_lls(rng, true, fit, x0, data, n,savefilename), 1:20)
