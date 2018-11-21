function [XS,LL] = LL_land_analysis_grad(ratname,model_type,sessid,save_pth)
%%

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');
load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');

try
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
catch
    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
end

load(fullfile(save_pth,sprintf('gML_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));

if ~exist('xf'); xf = history.x(:,end); end

%%

params = {'\sigma_i','inatt','B','\lambda','\sigma_a','\sigma_s','\phi','\tau_{\phi}'};
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

figure;set(gcf,'color','w');

ndxs = 10;

LL = NaN(ndxs,1);

dxs = linspace(-0.0000002,0.0000002,ndxs);

violate = NaN(numel(xf),ndxs);

for i = 1:ndxs
    
    x = xf + dxs(i) * g;
        
    violate(:,i) = x < lb | x > ub;
    
    x(x < lb) = lb(x<lb);
    x(x > ub) = ub(x>ub);
    
    XS(i,:) = x;
    
    LL(i) = LL_all_trials(x,data,dt,n,nstdc,dimz,dimd,dimy,...
        use,N,true,fr_func,[],x0(~fit_vec),fit_vec);
    
    subplot(2,1,1);
    hold on;
    plot(dxs(i),LL(i),'kx');
    ylabel('-LL');xlabel('frac of g added to x_{ML}');
    set(gca,'box','off');
    drawnow;
    
end

subplot(2,1,2);imagesc(dxs,1:numel(x),violate);
drawnow;

