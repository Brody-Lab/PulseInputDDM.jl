clearvars;

global N dt n dimz dimd dimy fr_func

load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/julia/spikes/157201_157357_157507_168499/17892933/17901706/17904427/julia_T036_157201_157357_157507_168499_spikes.mat')
load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/new2/157201_157357_157507_168499/data_T036_157201_157357_157507_168499_spikes.mat')

n = 203; dt = 2e-2; dimz = 8; dimd = 1; dimy = 4;

fr_func = @(x,z)bsxfun(@plus,x(:,1),bsxfun(@rdivide,x(:,2),...
    (1 + exp(bsxfun(@plus,-x(:,3) * z,x(:,4))))))' + eps;
N = max(data(numel(data)).N);

dtFP = dt; %save binning dt

fit_vec = true(numel(x0),1); fit_vec(1:2) = false; fit_vec(7:8) = false;

xgen = x0; xgen(fit_vec) = xf;

model_type = 'spikes';

[posterior,xc] = LL_all_trials_v2(xgen,data,dtFP,n,dimz,dimd,dimy,...
    model_type,N,fr_func,true);
LLstar = LL_all_trials_v2(xgen,data,dtFP,n,dimz,dimd,dimy,...
    model_type,N,fr_func,false);

%%

figure(1);clf;plot(posterior{12}(:,end))