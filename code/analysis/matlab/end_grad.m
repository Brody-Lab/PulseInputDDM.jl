clearvars


load('/Users/briandepasquale/Dropbox/results/new2/157201_157357_157507_168499/T036_157201_157357_157507_168499_spikes.mat')
load('/Users/briandepasquale/Dropbox/results/new2/157201_157357_157507_168499/data_T036_157201_157357_157507_168499_spikes.mat')

%%

dx = 1e-6;
g = zeros(numel(xf),1);

for i = 1:numel(xf)
    
    disp(i);
    
    x = xf;
    x(i) = x(i) + dx/2;
    LL2 = LL_all_trials_v2(x,data,dt,n,dimz,dimd,dimy,use,N,fr_func,false,x0(~fit_vec),fit_vec);
    
    x = xf;
    x(i) = x(i) - dx/2;
    LL1 = LL_all_trials_v2(x,data,dt,n,dimz,dimd,dimy,use,N,fr_func,false,x0(~fit_vec),fit_vec);
    
    g(i) = (LL2 - LL1)/dx;
    
end