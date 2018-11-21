function [xf,xy,mtcurve,CI,R2] = post_fit_params_and_good_fit(ratname,model_type,sessid,in_pth,save_pth)

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');

try
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
catch
    try
        load(fullfile(save_pth,sprintf('julia_history_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
        xf = history(:,end);
    catch
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
        if ~exist('xf'); xf = history.x(:,end); end
        try
            load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
        end
    end
    
end
%save binning dt
dtFP = dt;
%%

CI = NaN(numel(xf),1);

if exist('H')
    
    %blah = 3:6;
    blah = [];
    tol = (ub(blah) - lb(blah)) * 1e-3;
    
    bad = blah(((xf(blah) - lb(blah)) < tol) | ((ub(blah) - xf(blah)) < tol));
    
    H(bad,:) = []; H(:,bad) = [];
    
    Hvec = fit_vec;
    Hvec2 = Hvec(fit_vec);
    Hvec2(bad) = false;
    
    %if all(eig(H) > 0)
    CI(Hvec2) = real(2*sqrt(diag(inv(H))));
    %end
    
end

%%

xgen = x0;
xgen(fit_vec) = xf;

[data] = load_data(ratname,model_type,sessid,in_pth,dtFP);

LLstar = LL_all_trials(xf,data,dtFP,n,nstdc,dimz,dimd,dimy,...
    use,N,settle,fr_func,false,x0(~fit_vec),fit_vec);

for i = 1:2
    [fake_data] = generate_data(ratname,use,sessid,in_pth,xgen,[],[],dtFP);
    LL(i) = LL_all_trials(xf,fake_data,dtFP,n,nstdc,dimz,dimd,dimy,...
        use,N,settle,fr_func,false,x0(~fit_vec),fit_vec);
end

R2 = 1 - abs((mean(LL)-LLstar))/LLstar;

%%

[data] = load_data(ratname,model_type,sessid,in_pth,dtFP);
%compute posterior
[~,xc] = LL_all_trials(xf,data,dtFP,n,nstdc,dimz,dimd,dimy,...
    use,N,settle,fr_func,true,x0(~fit_vec),fit_vec);

%%

dimz2 = sum(fit_vec(1:dimz)); %how many parameters were for the latent
xy = reshape(xf(dimz2+1:end),N,dimy);
mtcurve = NaN(N,n);

for i = 1:N
        
    switch dimy
        case 4
            temp = fr_func([xy(i,1:2),abs(xy(i,3)),xy(i,4)],xc);
        case 2
            temp = fr_func([abs(xy(i,1)),xy(i,2)],xc);
    end
    mtcurve(i,:) = (temp - min(temp))/max((temp - min(temp)));
  
end

