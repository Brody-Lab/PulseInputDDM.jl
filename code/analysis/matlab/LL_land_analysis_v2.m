function [LL,dd,V,H,params] = LL_land_analysis_v2(ratname,model_type,sessid,save_pth,julia)
%%

if nargin < 5 || isempty(julia)
    julia = false;
end

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');
load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');

%just to get N, n, dt etc.
load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/new2/157201_157357_157507_168499/T036_157201_157357_157507_168499_spikes.mat','-regexp', '^(?!(xf|history)$).')
fit_vec([7,8]) = false;

if julia   
    try
        load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
        %xf = deparameterize(x0,model_type,N,xf,fit_vec);
        xf = xf(fit_vec);
    catch
        
    end
    
else    
    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
    try
        load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    end
    
end

x0 = deparameterize(x0,model_type,N,xf,fit_vec);
xf = x0(fit_vec);
%LLstar = ll_wrapper(xf,x0,fit_vec,data,model_type,dt,n,dimz,dimd,dimy,N,fr_func);

%%

params = {'\sigma_i','inatt','B','\lambda','\sigma_a','\sigma_s','\phi','\tau_{\phi}'};
if (strcmp(model_type,'joint') || strcmp(model_type,'choice')); params = cat(2,params,'bias'); end;
for i = 1:4
    switch i
        case 1
            blah = 'a';
        case 2
            blah = 'b';
        case 3
            blah = 'c';
        case 4
            blah = 'd';
    end
    for j = 1:N
        params = cat(2,params,sprintf('%s%g',blah,j));
    end
end

%%

[V,d] = eig(H);
dd = diag(d);
badEV = find(dd < 0);

if ~isempty(badEV)
    
    %EVs = 1:rank(H);
    %EVs = unique([1,5,find(dd < 0)',rank(H)]);
    %EVs = unique([1,badEV',rank(H):rank(H)]);
    EVs = [badEV'];
    
    ndxs = 71;
    
    LL = NaN(ndxs,numel(EVs));
    
    for j = 1:numel(EVs)
                
        dxs = linspace(-0.1,0.1,ndxs);
        
        figure(j);clf;set(gcf,'color','w');
        
        for i = 1:ndxs
            
            x = xf + dxs(i) * V(:,EVs(j));
            
            LL(i,j) = ll_wrapper(x,x0,fit_vec,data,model_type,dt,n,dimz,dimd,dimy,N,fr_func);
            
            subplot(1,1,1);
            hold on;
            plot(dxs(i),LL(i,j),'kx');
            ylabel('-LL');xlabel('frac of EV added to x_{ML}');
            title(sprintf('EV %g',EVs(j)));
            set(gca,'box','off');
            drawnow;
            
        end
                
    end 
    
end
