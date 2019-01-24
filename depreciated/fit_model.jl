
addprocs(48);

@everywhere using do_maximum_likelihood, initialize_spike_obs_model
using global_functions, Pandas, JLD, initialize_latent_model

#ratname = "T036"
#sessid = [154154,154291,154448,154991,155124,155247,155840,157201,157357,157507,168499,168627];
ratname = "T035";
sessid = [152177,153274,153536,154288,154440,155839,156150,156443,156896,
            157200,157359,161394,162142, 162258,163098,163885,164449,164604,164752,164900,165058,166135,
            166590,167725,167855,167993,168132,168628,169448,169736,169873,169993]

model_type_dir = "spikes"
path=ENV["HOME"]*"/Dropbox/spike-data_latent-accum";
model_type = model_type_dir
map_str = "exp";
dt = 2e-2;
noise = "Poisson";
f_str = "exp";
#f_str = "sig";

data_n_1ms,N = make_data(path,sessid,ratname;dt=1e-3,organize="by_neuron");
data_t_20ms, = make_data(path,sessid,ratname;dt=dt,organize="by_trial");

#### compute linear regression slope of tuning to $\Delta_{LR}$ and miniumum firing based on binning and averaging

ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=1e-3),data_n_1ms["nT"],data_n_1ms["leftbups"],data_n_1ms["rightbups"]);
nconds = 7;

conds_bins = map(x->qcut(vcat(ΔLR[x]...),nconds,labels=false,duplicates="drop",retbins=true),data_n_1ms["trial"]);
fr1 = map(j -> map(i -> (1/1e-3)*mean(vcat(data_n_1ms["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N);

c0 = map((trial,k)->linreg(vcat(ΔLR[trial]...), vcat(k...)),data_n_1ms["trial"],data_n_1ms["spike_counts"]);

if f_str == "exp"
    py0 = map((x,c0)->vcat(minimum(x),c0[2]),fr1,c0);
elseif f_str == "sig"
    py0 = map((x,c0)->vcat(minimum(x),maximum(x)-minimum(x),c0[2],0.),fr1,c0);
end

#### compute the strength of priors for each neuron
Betas = logspace(-16,-2,15);
min_Beta = pmap(i->indmin(map(Beta->mean(map(j->cross_validate_ΔLRmodel(copy(py0[i]),data_n_1ms["spike_counts"][i],
            data_n_1ms["trial"][i],ΔLR,Int(ceil(0.90*length(data_n_1ms["spike_counts"][i]))),
            Beta,f_str,rng=j,dt=1e-3),1:10)),Betas)),1:N);

#### compute tuning curve parameters, given the priors
#betas = map(x->x*ones(4),Betas[min_Beta]);
#betas = vcat(1e-3.*ones(2),1e-1*ones(2));
betas = map(x->x*ones(length(py0[1])),Betas[min_Beta]);
mu0 = py0;
py1 = do_ML_spikes_ΔLR(deepcopy(py0),data_n_1ms,map_str,betas,mu0,f_str,dt=1e-3);

#### make fake data
#pzstar = [1e-6,      0.,         20.,        -1.,         40.,            1.,      1.,    0.2];
#kwargs = ((:py,py1))
#sampled_dataset!(data_t_20ms, pzstar, f_str; rng = 2, num_reps=1, noise=noise, kwargs);

#### fit simple latent model
pz1 = [1e-6,      0.,         10.,        -1,         10,            0.25,      1.,    0.2];
fit_vec_z = [falses(2);    trues(2); falses(4)];
fit_vec = cat(1,fit_vec_z,trues(length(py1[1])*N));
mu0 = py1;
pz2, py2 = do_ML_filt_bound(copy(pz1),deepcopy(py1),fit_vec,model_type,map_str,dt,data_t_20ms,betas,mu0,f_str)

#### fit full latent model
fit_vec_z = [falses(2);    trues(4); falses(2)];
fit_vec = cat(1,fit_vec_z,trues(length(py2[1])*N));
mu0 = py2;
pz3, py3 = do_ML_spikes(copy(pz2),deepcopy(py2),fit_vec,model_type,map_str,dt,data_t_20ms,betas,mu0,noise,f_str)

@save path*"/"*ratname*"_"*f_str*"_params.jld"
