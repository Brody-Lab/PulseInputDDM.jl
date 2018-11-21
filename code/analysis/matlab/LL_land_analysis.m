function LL_land_analysis(ratname,model_type,sessid,save_pth)
%%

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');
load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');

try
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
catch
    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
    try
        load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    end
    
end

if ~exist('xf'); xf = history.x(:,end); end

%%

CI = NaN(numel(xf),1);

if exist('H')
    
    %blah = 3:6;
    %blah = [1,3:4,6];
    %blah = [];
    %tol = (ub(blah) - lb(blah)) * 1e-2;
    %tol = [0.2,1,1e-1,2e-2]';
    %bad = blah(abs(xf(blah) - lb(blah)) < tol);
    bad = [];
    
    %bad = blah(((xf(blah) - lb(blah)) < tol) | ((ub(blah) - xf(blah)) < tol));
    
    H(bad,:) = []; H(:,bad) = [];
    
    Hvec = fit_vec;
    Hvec2 = Hvec(fit_vec);
    Hvec2(bad) = false;
    
    if all(eig(H) > 0)
        CI(Hvec2) = (2*sqrt(diag(inv(H))));
    end
    
end

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
    
    figure(100);set(100,'color','w');
    params2 = params(Hvec);
    params3 = params2(Hvec2);
    
        for i = 1:numel(badEV)
            subplot(2,ceil(numel(badEV)/2),i);
            plot(V(:,badEV(i)),'kx','Linestyle','none');
            set(gca,'XTick',[1:numel(params3)],'XTickLabels',params3);
            set(gca,'XTickLabelRotation',90);
            title(sprintf('sess %s, EV %g',strjoin(sessid,'_'),badEV(i)));
        end
        drawnow;
    
    %%
    
    %EVs = 1:rank(H);
    %EVs = unique([1,5,find(dd < 0)',rank(H)]);
    %EVs = unique([1,badEV',rank(H):rank(H)]);
    EVs = [badEV'];
    
    ndxs = 20;
    
    %ub2 = ub(Hvec2);
    %lb2 = lb(Hvec2);
    xf2 = xf(Hvec2);
    
    LL = NaN(ndxs,numel(EVs));
    DXS = NaN(size(LL));
    
    if 1
                
        for j = 1:numel(EVs)
            
            EV = V(:,EVs(j));
            
            %up = EV > 0;
            %down = ~up;
               
            %lpp = max([max([(ub2(down)-xf2(down))./EV(down);(lb2(up)-xf2(up))./EV(up)])]);
            %upp = min([min([(ub2(up)-xf2(up))./EV(up);(lb2(down)-xf2(down))./EV(down)])]);
            %dxs = linspace(lpp, upp, ndxs);
            %dxs = linspace(-sqrt(dd(EVs(j))),sqrt(dd(EVs(j))),ndxs);
            %dxs = linspace(-0.2048382,-0.2048381,ndxs);
            dxs = linspace(-1,1,ndxs);
            DXS(:,j) = dxs;
            violate = NaN(numel(xf),ndxs);
            
            figure(j);clf;set(gcf,'color','w');
            
            for i = 1:ndxs
                
                x = xf;
                x(Hvec2) = xf2 + dxs(i) * EV;
                
                violate(:,i) = x < lb | x > ub;
                
                x(x < lb) = lb(x<lb);
                x(x > ub) = ub(x>ub);
                %settle = true;
                
                LL(i,j) = LL_all_trials_v2(x,data,dt,n,dimz,dimd,dimy,...
                    use,N,fr_func,[],x0(~fit_vec),fit_vec);
                
                subplot(2,1,1);
                hold on;
                if any(EVs(j) == badEV)
                    plot(dxs(i),LL(i,j),'rx');
                else
                    plot(dxs(i),LL(i,j),'kx');
                end
                %plot(zeros(100,1),linspace(min(LL(:,j)),max(LL(:,j)),100),'k--');
                ylabel('-LL');xlabel('frac of EV added to x_{ML}');
                title(sprintf('EV %g',EVs(j)));
                set(gca,'box','off');
                drawnow;
                
            end
            %
            %
            %
            %         XF = bsxfun(@plus,xf(Hvec2),bsxfun(@times,EV,dxs));
            %
            %         params4 = params3(abs(EV) > 0.1);
            %         vec = find(abs(EV) > 0.1);
            %
            %         figure;
            %
            %         for i = 1:numel(params4)
            %             subplot(2,ceil(numel(params4)/2),i);
            %             plot(XF(vec(i),:));
            %             title(sprintf('%s',params4{i}));
            %             set(gca,'fontsize',12);
            %         end
            %         drawnow;
            
            subplot(2,1,2);imagesc(dxs,1:numel(x),violate);
            drawnow;
            
        end
        
    end
    
    %x = xf;
    %x(Hvec2) = xf2 + dxs(find(LL == min(LL))) * EV;
    %x0new = x;
    %save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'x0new','-append');
    
end
