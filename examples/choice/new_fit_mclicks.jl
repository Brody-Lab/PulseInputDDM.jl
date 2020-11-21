# # Loading data and fitting a choice model
using pulse_input_DDM, Flatten, MAT, Random, JLD2
import pulse_input_DDM: P_goright
num_array = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
src_data_folder="/scratch/ejdennis/d081_rawdata/";
fclicks_files=readdir(src_data_folder);
filename=fclicks_files[num_array]
string(src_data_folder,filename)
data = load_choice_data(string(src_data_folder,filename));
n = 53
fit = vcat(trues(9))
lb = vcat([0., 1., -5., 0., 0., 0.001, 0.0000001], [-30, 0.])
ub = vcat([20., 100., 5., 200., 10., 1.2, 1.], [30, 1.])
save_file_location = "/scratch/ejdennis/ddm_runs/"
xlb = vcat([0.,1.,-1.,0.,0.,0.1,0.001],[-3,0.])
xub = vcat([5.,30.,1.,10.,4.,1.05,0.1],[3,0.4])
x0 = vcat([0.1,15.,-0.1,5., 0.5, 0.02, 0.008], [0.,0.1])

savefilename = string(save_file_location,"/",filename[1:end-4],num_array)

function fit_x0(rng,cross, fit, x0, xlb, xub, lb, ub, data,n,savefilename)
    println("filename: ",filename)
    println("and rng: ",rng)
    x00 = xlb + (xub - xlb) .* rand(length(x0))
    x00[2]=15.
    options = choiceoptions(fit=fit, lb=lb,ub=ub) 
    model, output = optimize(data,options;cross=cross,x0=x00,extended_trace=true,show_trace=false,scaled=false)
    H = Hessian(model)
    CI, = CIs(H)
    ll = loglikelihood(model)
    println("ll: ",ll)
    save_choice_model(string(savefilename,"_",rng,"_",cross,"_choicemodel_results.mat"),model,options,CI)
    io=open(string(savefilename,"_",cross,"_ll",rng,".txt"),"w")
    println(io, string(ll))
    close(io)
end

output = map(rng -> fit_x0(rng, false, fit, x0, xlb, xub, lb, ub, data, n,savefilename), 1:20)
# trace = map(y -> hcat(map(x -> x.metadata["x"], y[2].trace)...), output)
outputt = map(rng -> fit_x0(rng, true, fit, x0, xlb, xub, lb, ub, data, n,savefilename), 1:20)
# tracet = map(y -> hcat(map(x -> x.metadata["x"], y[2].trace)...), output)

