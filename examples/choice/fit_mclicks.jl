# # Loading data and fitting a choice model
using pulse_input_DDM, Flatten
num_array = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
src_data_folder="/scratch/ejdennis/d081_rawdata/";
fclicks_files=readdir(src_data_folder);
filename=fclicks_files[num_array]
string(src_data_folder,filename)
data = load_choice_data(string(src_data_folder,filename));
n = 53
options = choiceoptions(fit = vcat(trues(9)),
    lb = vcat([0., 1., -5., 0., 0., 0.001, 0.005], [-30, 0.]),
    ub = vcat([2., 30., 5., 200., 10., 1.2, 1.], [30, 1.]))
save_file_location = "/scratch/ejdennis/ddm_runs/"

model, = optimize(data, options)
H = Hessian(model)
CI, = CIs(H);
ll = loglikelihood(model)
println("ll: ",ll)
save_choice_model(string(save_file_location,filename[1:end-4],num_array,"_results.mat"), model, options, CI)
save(string(save_file_location,filename[1:end-4],num_array,"_ll.mat"),ll)

