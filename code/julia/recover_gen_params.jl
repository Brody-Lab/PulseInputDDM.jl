
addprocs(48);

@everywhere using do_maximum_likelihood
using PyPlot, initialize_latent_model
using global_functions, Pandas, initialize_spike_obs_model
using StatsBase, JLD
@everywhere using sample_model

ratname = "T036"
model_type_dir = "spikes"
path=ENV["HOME"]*"/Dropbox/spike-data_latent-accum";
model_type = model_type_dir
map_str = "exp";
sessid = [157201];

f_str = "softplus";
#f_str = "sig";
num_reps = 1;
noise = "Poisson";
#noise = "Gaussian";
#          vari       inatt        B          lambda       vara          vars       phi    tau_phi 
pzstar = [1e-6,      0.,         20.,        -1.,         40.,            1.,         1.0,    0.2];
pz0 =    [1e-6,      0.,         5.,         -2.,         10.,            0.25,       1.0,    0.2];
fit_vec_z = [falses(2);    trues(4); falses(2)];

#pyp = [[1.,10.,1.,0.],[1.,10.,0.1,0.]];
py = [[1.,10.,0.],[1.,-10.,0.]];
py = [py;deepcopy(py);deepcopy(py);deepcopy(py);deepcopy(py)];
py = [py;deepcopy(py)];
py = [py;deepcopy(py)];

dt = 1e-3;
data, = make_data(path,sessid,ratname;dt=dt,organize="by_trial");

N = 40;
pz3, CI_z_plus, CI_z_minus, py3, data_t_1ms = make_data_and_fit(deepcopy(data),num_reps,
        fit_vec_z,dt,copy(pzstar),pz0,model_type,map_str,f_str,
        betas=map(i->zeros(length(py[1])),1:N),pystar=deepcopy(py[1:N]),
        all_sim=true,N=N,noise=noise,fity=true,fit_lambda=false,n=53,
        x_tol=1e-12,f_tol=1e-4,g_tol=1e-2);

results = cat(2,CI_z_minus[fit_vec_z],pz3[fit_vec_z],CI_z_plus[fit_vec_z])

pz = cat(2,results[:,2]...)
CIminus = abs.(cat(2,results[:,1]...) - pz)
CIplus = abs.(cat(2,results[:,3]...) - pz);

CIplus[isinf.(CIplus)] = 1e8;

@save path*"/"*ratname*"_"*f_str*"_"*string(length(sessid))*"_sessions"*"_"*string(sum(fit_vec_z))*"_simulated_data_params.jld"

if false
    num_rows, num_cols = 1, 4
    fig, ax = subplots(num_rows, num_cols, figsize=(16,6))
    subplot_num = 0
    param_str = ["B",L"\lambda",L"\sigma_a^2",L"\sigma_s^2"]

    for j in 1:num_cols
        subplot_num += 1
        ax[j][:errorbar](1,pz[subplot_num], yerr = cat(2,CIminus[subplot_num],CIplus[subplot_num])',fmt="o")
        ax[j][:plot](linspace(0,maximum(2),100),pzstar[fit_vec_z][subplot_num]*ones(100))
        ax[j][:set_title]("$(param_str[subplot_num])",fontsize=10)
        ax[j][:set_xlabel]("$(length(data["T"])) trials, # of neurons",fontsize=10)

        if param_str[subplot_num] == L"\sigma_a^2"
            ax[j][:set_ylim](0, 300)
        elseif param_str[subplot_num] == L"\sigma_s^2"
            ax[j][:set_ylim](0, 5)
        elseif param_str[subplot_num] == "B"
            ax[j][:set_ylim](0, 70)
        end
    end
end
