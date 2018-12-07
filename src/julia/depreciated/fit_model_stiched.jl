
addprocs(48);

@everywhere using do_maximum_likelihood, initialize_spike_obs_model
using global_functions, Pandas, JLD, initialize_latent_model, MAT

model_type_dir = "spikes"
path=ENV["HOME"]*"/Dropbox/spike-data_latent-accum";
model_type = model_type_dir
map_str = "exp";
noise = "Poisson";
f_str = "softplus";
fit_vec_z = [trues(1);falses(1);trues(4); falses(2)];

#@load path*"/PPC_shifted_stiched_data.jld" data py1 py0 dt N data_spike_counts_n
@load path*"/data/results/julia/stitched/PPC_shifted_stiched_data.jld" data py1 py0 dt N data_spike_counts_n
#@load path*"/data/results/julia/stitched/FOF_shifted_stiched_data.jld" data py1 py0 dt N data_spike_counts_n
#@load path*"/STR_shifted_stiched_data.jld" data py1 py0 dt N data_spike_counts_n

#@load path*"/STR_stiched_data.jld" data py1 py0 dt N data_spike_counts_n
#@load path*"/FOF_stiched_data.jld" data py1 py0 dt N data_spike_counts_n
#@load path*"/PPC_stiched_data.jld" data py1 py0 dt N data_spike_counts_n

betas = map(x->x*ones(length(py0[1])),zeros(N));

#### fit simple latent model
#pz1 = [1.,      0.,         10.,        -5,         10,            0.25,      1.0,    0.2];
#fit_vec_bound = [falses(2);trues(2);falses(4)];
#fit_vec = cat(1,fit_vec_bound,trues(length(py1[1])*N));
#mu0 = py1;
#pz2, py2 = do_ML_filt_bound(copy(pz1),deepcopy(py1),fit_vec,model_type,map_str,dt,data,betas,mu0,f_str,noise)

#@save path*"/FOF_stiched_results.jld"
#@save path*"/PPC_stiched_results.jld"
#@save path*"/STR_stiched_results.jld"
#@save path*"/PPC_shifted_stiched_results.jld"
#@save path*"/FOF_shifted_stiched_results.jld"
#@load path*"/old_fits/stitched/FOF_shifted_stiched_results.jld" py2 pz2
@load path*"/old_fits/stitched/PPC_shifted_stiched_results.jld" py2 pz2
#@save path*"/STR_shifted_stiched_results.jld"

#### fit full latent model
fit_vec = cat(1,fit_vec_z,trues(length(py2[1])*N));
mu0 = py2;
pz3, py3 = do_ML_spikes(copy(pz2),deepcopy(py2),fit_vec,model_type,map_str,dt,data,betas,mu0,noise,f_str,n=53)

#@save path*"/FOF_stiched_results.jld"
#@save path*"/PPC_stiched_results.jld"
#@save path*"/STR_stiched_results.jld"
#@save path*"/PPC_shifted_stiched_results.jld"
#@save path*"/FOF_shifted_stiched_results.jld"
@save path*"/PPC_shifted_stiched_5param_results.jld"
#@save path*"/FOF_shifted_stiched_5param_results.jld"
#@save path*"/STR_shifted_stiched_results.jld"
