clearvars;

sessid = '157201 157357 157507 168499';
sessid = strsplit(sessid);
ratname = 'T036';
model_type = 'spikes';
save_pth = '/Users/briandepasquale/Documents/Dropbox/results/multiple_session/T036_4sess__6_4_param_1e_32/';

load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid),model_type)));
load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid),model_type)));

num_fitz = sum(fit_vec(1:dimz));

params = x0;
params(fit_vec) = xf;

%      vari     inatt     B       lambda    vara    vars     phi    tau_phi 
Hlb =  [1e-1    1e-2      1e-1    NaN       1e-1    1e-1     1e-2    1e-3]';
Hub =  [inf     1 - 1e-2  inf     NaN       inf     inf      inf     inf]';

Hlby = reshape(repmat([1e-1,NaN,NaN,-10]',1,N)',[],1);
Hlb = cat(1,Hlb,Hlby);

Huby = reshape(repmat([inf,NaN,NaN,10]',1,N)',[],1);
Hub = cat(1,Hub,Huby);

x = params(1:dimz);
xy = reshape(params(dimz+1:end),N,dimy);

lowfr = xy(:,1) < 1;
badz = x(fit_vec(1:dimz)) - 1e-2;

bad = find((params(fit_vec) < Hlb(fit_vec)) | (params(fit_vec) > Hub(fit_vec)));

XF = zeros(9,N);

for n = 1:N
    
    %vec = setdiff(1:numel(xf),bad);
    vec = [1:3,5,6,num_fitz+1+(n-1):N:numel(xf)];
    XF(:,n) = xf(vec);
    
    CI(:,n) = 2*sqrt(diag(inv(H(vec,vec))));
    d(:,n) = eig(H(vec,vec));
        
end
