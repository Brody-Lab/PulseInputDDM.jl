
addprocs(48);

@everywhere using do_maximum_likelihood
using PyPlot, initialize_latent_model
using global_functions, Pandas, JLD, latent_model
@everywhere using sample_model,initialize_spike_obs_model

ratname = "T036"
model_type_dir = "spikes"
path=ENV["HOME"]*"/Dropbox/spike-data_latent-accum";
model_type = model_type_dir
map_str = "exp";
#sessid = [154154];
#sessid = [157201];
#sessid = [157507,168499,168627];
#sessid = [157201,157357,157507,168499];
sessid = [154154,154291,154448,154991,155124,155247,155840,157201,157357,157507,168499,168627];
dt = 2e-2;
noise = "Poisson";

data_n_1ms,N = make_data(path,sessid,ratname;dt=1e-3,organize="by_neuron");

ΔLR = map((x,y,z)->diffLR(x,y,z,path=true,dt=1e-3),data_n_1ms["nT"],data_n_1ms["leftbups"],data_n_1ms["rightbups"]);
nconds = 7;

conds_bins = map(x->qcut(vcat(ΔLR[x]...),nconds,labels=false,duplicates="drop",retbins=true),data_n_1ms["trial"]);
fr1 = map(j -> map(i -> (1/1e-3)*mean(vcat(data_n_1ms["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N); #this doesn't work anymore

c0 = map((trial,k)->linreg(vcat(ΔLR[trial]...), vcat(k...)),data_n_1ms["trial"],data_n_1ms["spike_counts"]);

Betas = logspace(-16,-2,15);

py0 = map((x,c0)->vcat(minimum(x),c0[2]),fr1,c0);
f_str = "exp";

f_str = "sig";
py0 = map((x,c0)->vcat(minimum(x),maximum(x)-minimum(x),c0[2],0.),fr1,c0);

min_Beta = pmap(i->indmin(map(Beta->mean(map(j->cross_validate_ΔLRmodel(copy(py0[i]),data_n_1ms["spike_counts"][i],
            data_n_1ms["trial"][i],ΔLR,Int(ceil(0.90*length(data_n_1ms["spike_counts"][i]))),
            Beta,f_str,rng=j,dt=1e-3),1:10)),Betas)),1:N);

#betas = map(x->x*ones(4),Betas[min_Beta]);
betas = map(x->x*ones(length(py0[1])),Betas[min_Beta]);
#betas = vcat(1e-3.*ones(2),1e-1*ones(2));

mu0 = py0;
py1 = do_ML_spikes_ΔLR(deepcopy(py0),data_n_1ms,map_str,betas,mu0,f_str,dt=1e-3);

py1

if true

    fig, ax = subplots(1,3, figsize=(12,4))
    subplot_num = 0;

    for i = 1:3
        for j = 1:1
        subplot_num += 1

        xc = conds_bins[subplot_num][2][1:end-1] + diff(conds_bins[subplot_num][2])/2;

        ax[i,j][:scatter](xc,fr1[subplot_num],color="grey",label="data")
        
        if f_str == "sig"
            ax[i,j][:plot](xc,my_sigmoid(xc,py1[subplot_num]),color="blue",label="fit")
        elseif f_str == "exp"
            ax[i,j][:plot](xc,my_exp(xc,py1[subplot_num]),color="red",label="fit")
        end

        ax[i,j][:set_ylabel]("rate")
        ax[i,j][:set_xlabel](L"\Delta_{LR}(T)")
        ax[i,j][:set_xlim](-40,40);
        ax[i,j][:spines]["top"][:set_color]("none") # Remove the top axis boundary
        ax[i,j][:spines]["right"][:set_color]("none") # Remove the top axis boundary

        end
    end
    #savefig(pth*"/figures/labmeeting/"*ratname*"_"*model_type_dir*"_tanh_psychometric.png")
    
end

data_t_20ms, = make_data(path,sessid,ratname;dt=dt,organize="by_trial");

#pzstar = [1e-6,      0.,         20.,        -1.,         40.,            1.,      1.,    0.2];
#kwargs = ((:py,py1))
#sampled_dataset!(data_t_20ms, pzstar, f_str; rng = 2, num_reps=1, noise=noise, kwargs);

pz1 = [1e-6,      0.,         10.,        -1,         10,            0.25,      1.,    0.2];
fit_vec_z = [falses(2);    trues(2); falses(4)];
fit_vec = cat(1,fit_vec_z,trues(length(py1[1])*N));
mu0 = py1;

pz2, py2 = do_ML_filt_bound(copy(pz1),deepcopy(py1),fit_vec,model_type,map_str,dt,data_t_20ms,betas,mu0,f_str)

cat(2,pz1,pz2)

cat(2,py0,py1,py2)

output = map((x,y,z)->binLR(x,y,z,path=true,dt=1e-3),data_n_1ms["nT"],data_n_1ms["leftbups"],data_n_1ms["rightbups"]);

L = map(x->x[1],output);
R = map(x->x[2],output);

A = pmap((T,L,R)->filter_bound(pz2[4],pz2[3],T,L,R;dt=1e-3),data_n_1ms["nT"],L,R);

map(plot,A);

conds_bins = map(x->qcut(vcat(A[x]...),nconds,labels=false,duplicates="drop",retbins=true),data_n_1ms["trial"]);
fr2 = map(j -> map(i -> (1/1e-3)*mean(vcat(data_n_1ms["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N); #this doesn't work anymore

if true

    fig, ax = subplots(1,3, figsize=(12,4))
    subplot_num = 0;

    for i = 1:3
        for j = 1:1
        subplot_num += 1

        xc = conds_bins[subplot_num][2][1:end-1] + diff(conds_bins[subplot_num][2])/2;

        ax[i,j][:scatter](xc,fr2[subplot_num],color="grey",label="data")
        if f_str == "sig"
            ax[i,j][:plot](xc,my_sigmoid(xc,py2[subplot_num]),color="blue",label="fit")
        elseif f_str == "exp"
            ax[i,j][:plot](xc,my_exp(xc,py2[subplot_num]),color="red",label="fit")
        end

        ax[i,j][:set_ylabel]("rate")
        ax[i,j][:spines]["top"][:set_color]("none") # Remove the top axis boundary
        ax[i,j][:spines]["right"][:set_color]("none") # Remove the top axis boundary
        #ax[i][:legend](loc="best")

        end
    end
    #savefig(pth*"/figures/labmeeting/"*ratname*"_"*model_type_dir*"_tanh_psychometric.png")
    
end

fit_vec_z = [falses(2);    trues(4); falses(2)];
fit_vec = cat(1,fit_vec_z,trues(length(py2[1])*N));
mu0 = py2;

#fit full model
pz3, py3 = do_ML_spikes(copy(pz2),deepcopy(py2),fit_vec,model_type,map_str,dt,data_t_20ms,betas,mu0,noise,f_str)

cat(2,pz1,pz2,pz3)

cat(2,py1,py2,py3)

A = pmap((T,L,R) -> sample_latent(T,L,R,pz3),data_n_1ms["T"],data_n_1ms["leftbups"],data_n_1ms["rightbups"]);

map(x->plot(x),A);

if f_str == "exp"
    lambda = map(py->map(a->my_exp(a,py),A),py3);
elseif f_str == "sig"
    lambda = map(py->map(a->my_sigmoid(a,py),A),py3);
end

map(x->plot(x),lambda[3]);

posterior = LL_all_trials(pz3, data_t_20ms, model_type, f_str; comp_posterior=true, n=53, py=py3);

xc, = bins(pz3[3],n=53);
μ_posterior = map(x->x'*xc,posterior);

map(plot,μ_posterior);

if f_str == "exp"
    lambda = map(py->map(a->my_exp(a,py),μ_posterior),py3);
elseif f_str == "sig"
    lambda = map(py->map(a->my_sigmoid(a,py),μ_posterior),py3);
end

map(x->plot(x),lambda[3]);

#add dt input to LL_all_trials to get 1e-3 resolution
data_n_20ms, = make_data(path,sessid,ratname;dt=dt,organize="by_neuron");
conds_bins = map(x->qcut(vcat(μ_posterior[x]...),nconds,labels=false,duplicates="drop",retbins=true),data_n_20ms["trial"]);
fr3 = map(j -> map(i -> (1/dt)*mean(vcat(data_n_20ms["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N); #this doesn't work anymore

if true

    fig, ax = subplots(5,2, figsize=(12,24))
    subplot_num = 0;

    for i = 1:5
        for j = 1:2
        subplot_num += 1

        xc = conds_bins[subplot_num][2][1:end-1] + diff(conds_bins[subplot_num][2])/2;

        ax[i,j][:scatter](xc,fr3[subplot_num],color="grey",label="data")
        if f_str == "sig"
            ax[i,j][:plot](xc,my_sigmoid(xc,py3[subplot_num]),color="blue",label="fit")
        elseif f_str == "exp"
            ax[i,j][:plot](xc,my_exp(xc,py3[subplot_num]),color="red",label="fit")
        end

        ax[i,j][:set_ylabel]("rate")
        ax[i,j][:spines]["top"][:set_color]("none") # Remove the top axis boundary
        ax[i,j][:spines]["right"][:set_color]("none") # Remove the top axis boundary
        #ax[i][:legend](loc="best")

        end
    end
    #savefig(pth*"/figures/labmeeting/"*ratname*"_"*model_type_dir*"_tanh_psychometric.png")
    
end

ΔLRT = map(diffLR,data_n_1ms["nT"],data_n_1ms["leftbups"],data_n_1ms["rightbups"]);
nconds = 3
conds = qcut(ΔLRT,nconds,labels=false)+1;

fr1 = map(j -> map(i -> (1/1e-3)*mean(vcat(data_n_1ms["spike_counts"][j]...)[conds_bins[j][1] .== i]),0:nconds-1),1:N); #this doesn't work anymore

num_rows, num_cols = 2,6
fig, ax = subplots(num_rows, num_cols, figsize=(24,12))
subplot_num = 0
colors = ["red","black","green"]

filtSD = 10

for i in 1:num_rows
    for j in 1:num_cols
        
        subplot_num += 1
        my_max = 0.;
        
        for k = 1:3
                                    
            #ax[i, j][:plot]((1:maximum(data["nT"]))*1e-3,
            #    FilterSpikes(nanmean(rates_sampled[subplot_num,:,(conds2 .== k)],2),filtSD),
            #    linestyle="-",color=colors[k])
            ax[i, j][:fill_between]((1:maximum(data["nT"]))*1e-3,
                vec(FilterSpikes(vec(nanmean(rate_mat[subplot_num,:,(conds .== k)],2) - 
                            nanstderr(rate_mat[subplot_num,:,(conds .== k)],2)),filtSD)),
                vec(FilterSpikes(vec(nanmean(rate_mat[subplot_num,:,(conds .== k)],2) + 
                            nanstderr(rate_mat[subplot_num,:,(conds .== k)],2)),filtSD)),
                alpha=0.3,color=colors[k])
            
            #my_max2 = maximum(FilterSpikes(filter(!isnan,vec(nanmean(rate_mat[subplot_num,1:500,(conds .== k)],2) + 
            #            nanstderr(rate_mat[subplot_num,1:500,(conds .== k)],2))),filtSD))
            #my_max = max(my_max,my_max2)
            
        end
        
        ax[i, j][:set_xlim]((0, 0.5))
        #ax[i, j][:set_ylim]((0.,my_max));
        ax[i, j][:set_xlabel]("time (s)")
        ax[i, j][:set_ylabel]("Firing rate (Hz)")
        ax[i, j][:set_title]("$subplot_num")
        ax[i, j][:spines]["top"][:set_color]("none") # Remove the top axis boundary
        ax[i, j][:spines]["right"][:set_color]("none") # Remove the top axis boundary  
        
    end
end
#savefig(pth*"/figures/labmeeting/"*ratname*"_"*model_type_dir*"_PSTHs.png")

@save path*"/"*ratname*"_"*"params_1neuron.jld"
