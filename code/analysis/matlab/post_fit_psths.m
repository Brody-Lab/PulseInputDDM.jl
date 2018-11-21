function [xf,xy,mtcurve,CI] = post_fit_psths(ratname,model_type,sessid,in_pth,save_pth)

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');

load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/new2/157201_157357_157507_168499/T036_157201_157357_157507_168499_spikes.mat','-regexp', '^(?!(xf|history)$).')
fit_vec([7,8]) = false;

try
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    xf = xf(fit_vec);
    
catch   
    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
    try
        load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    end
    
end

if ~exist('xf'); xf = history.x(:,end); end
%save binning dt
dtFP = dt;

%xf = bsxfun(@plus,lb(fit_vec),bsxfun(@times,(ub(fit_vec)-lb(fit_vec)).*0.5,(1+tanh(xf))));
%x0 = bsxfun(@plus,lb,bsxfun(@times,(ub-lb).*0.5,(1+tanh(x0))));

%%

%generate some fake data
%if ~exist('xgen')
xgen = x0;
xgen(fit_vec) = xf;
%end

% [data] = load_data(ratname,model_type,sessid,in_pth,dtFP);
% 
% LLstar = LL_all_trials(xf,data,dtFP,n,nstdc,dimz,dimd,dimy,...
%     use,N,settle,fr_func,false,x0(~fit_vec),fit_vec);
% 
% for i = 1:100
%     [fake_data] = generate_data(ratname,use,sessid,in_pth,xgen,[],[],dtFP);
%     LL(i) = LL_all_trials(xf,fake_data,dtFP,n,nstdc,dimz,dimd,dimy,...
%         use,N,settle,fr_func,false,x0(~fit_vec),fit_vec);
% end

%%

dt = 1e-3;
%dt2 = 5e-3;
%load data used to fit the model
%this will only load original data, meaning that this code cannto be used
%in situations where fake data was fit
%[data] = load_data(ratname,model_type,sessid,in_pth,dt2);
%compute psth and cta for the data
%[psth] = psth_cta(data,dt2,1);
%ccorr = compute_average_autcorr(psth,dt2);

%[fake_data] = generate_data(ratname,use,sessid,in_pth,xgen,true,1e4,dt2);
%[fake_psth] = psth_cta(fake_data,dt2,1);
%fake_ccorr = compute_average_autcorr(fake_psth,dt2);

[data] = load_data_v2(ratname,sessid,in_pth,dt,model_type);         %load and "package" the data
[fake_data] = generate_data_v2(ratname,model_type,sessid,in_pth,xgen,true,3e4,dt);
[psth,conds,cta] = psth_cta(data,dt,10);
[fake_psth,conds_fake] = psth_cta(fake_data,dt,10);

%%

[data] = load_data_v2(ratname,sessid,in_pth,dtFP,model_type);         %load and "package" the data
%compute posterior
% [posterior,xc] = LL_all_trials(xf,data,dtFP,n,nstdc,dimz,dimd,dimy,...
%     use,N,settle,fr_func,true,x0(~fit_vec),fit_vec);
% %calculate posterior tuning curve
% [mSC,xe2] = posterior_tuning_curve(xf,fit_vec,data,xc,posterior,dtFP);

%% forward model
% use2.choice = false; use2.spikes = false;
% forward_data = data;
% for i = 1:numel(forward_data)
%     forward_data(i).N = [];
% end
% [~,~,alpha] = LL_all_trials(xf,forward_data,dtFP,n,nstdc,dimz,dimd,dimy,...
%     use2,0,settle,fr_func,true,x0(~fit_vec),fit_vec);

%%

dimz2 = sum(fit_vec(1:dimz)); %how many parameters were for the latent
xy = reshape(xf(dimz2+1:end),N,dimy);

x = x0(1:dimz); %unpack parameters
x(logical(fit_vec(1:dimz))) = xf(1:dimz2);

n_conds = numel(unique(conds));
max_br = 1;
min_br = 0.4;
n_each_clr = floor(n_conds/2);
br_vec = linspace(min_br,max_br,n_each_clr);
for i=1:n_conds
    if i <= n_each_clr
        % reds for low conditions
        clrs(i,:) = [br_vec(i) 0 0];
        clrs2(i,:) = [0 0 br_vec(i)];
    elseif i > ((n_conds+1)/2)
        % greens for high conditions
        clrs(i,:) = [0 br_vec(end-i+n_conds-n_each_clr+1)  0];
        clrs2(i,:) = [0 br_vec(end-i+n_conds-n_each_clr+1)  0];
    else
        % black for middle condition is there is one
        clrs(i,:) = [0.7 0.7 0];
        clrs2(i,:) = [0 0.7 0.7];
    end
end

mtcurve = NaN(N,n);

if 1
    for i = 1:N
        
        figure;clf;set(gcf,'color','w');
        
%         subplot(1,1,1);hold on;
%         tcurve = fr_func(xy(i,:),xc);
%         %changed 2/28, to accomodate varioud fr_func, but didn't 100% check
%         %OK.
%         switch dimy
%             case 4
%                 temp = fr_func([xy(i,1:2),abs(xy(i,3)),xy(i,4)],xc);
%             case 2
%                 temp = fr_func([abs(xy(i,1)),xy(i,2)],xc);
%         end
%         mtcurve(i,:) = (temp - min(temp))/max((temp - min(temp)));
%         plot(xc,tcurve,'color','r','LineWidth',2);
%         scatter(xe2(1:end-1)+mean(diff(xe2))/2,mSC(:,i),'x','MarkerFaceColor','k',...
%             'MarkerEdgeColor','k');
%         set(gca,'box','off','ylim',[0 max(tcurve)]);
%         xlabel('post. mean');ylabel('fr (hz)');
%         title(sprintf('tuning curve wrt posterior, neuron %g',i));
        
%         subplot(1,2,2);hold on;
%         %plot(0:dt:0.5-dt,nanmean(cta(i,:,:),3),'k','Linestyle','--');
%         %plot(0:dt:0.5-dt,nanmean(fake_cta(i,:,:),3),'k','LineWidth',2);
%         nplot = 10;
%         nplot2 = 2;
%         %errorbar(0:nplot*dt:0.5-dt,nanmean(cta(i,1:nplot:end,:),3),nanstderr(cta(i,1:nplot:end,:),3),'r','Linestyle','-');
%         %errorbar(0:nplot*dt:0.5-dt,nanmean(fake_cta(i,1:nplot:end,:),3),nanstderr(fake_cta(i,1:nplot:end,:),3),'k','LineWidth',1);
%         legend('true','fake','Location','Best');legend boxoff;
%         xlabel('time (s)'); ylabel('Hz');
%         title(sprintf('click-triggered average, neuron %g',i));
%         set(gca,'xlim',[0 0.5]);
                
        %xcorrt = numel(squeeze(nanmean(ccorr(i,i,:,:),4))');
        %plot(-(xcorrt-1)/2:(xcorrt-1)/2,squeeze(nanmean(ccorr(i,i,:,:),4))');hold on;
        %plot(-(xcorrt-1)/2:(xcorrt-1)/2,squeeze(nanmean(fake_ccorr(i,i,:,:),4))');
        
%         if mod(i-1,3)+1 == 1
%             fh = figure;clf;set(gcf,'color','w');hold on;
%         end
        
%         for j = 1:N
%                         
%             if any(~isnan(squeeze(nanmean(ccorr(i,j,:,:),4))'))
%                 
%                 set(0,'CurrentFigure',fh);
%                 subplot(3,3,((mod(i-1,3)+1)-1)*3+mod(j-1,3)+1);hold on;
%                 
%                 temp1 = squeeze(nanmean(ccorr(i,j,:,:),4))';
%                 temp2 = squeeze(nanstderr(ccorr(i,j,:,:),4))';
%                 errorbar(dt2*(-(xcorrt-1)/2:nplot2:(xcorrt-1)/2),temp1(1:nplot2:end),temp2(1:nplot2:end),...
%                     'color','r','LineStyle','-');
%                 
%                 temp1 = squeeze(nanmean(fake_ccorr(i,j,:,:),4))';
%                 temp2 = squeeze(nanstderr(fake_ccorr(i,j,:,:),4))';
%                 errorbar(dt2*(-(xcorrt-1)/2:nplot2:(xcorrt-1)/2),temp1(1:nplot2:end),temp2(1:nplot2:end),...
%                     'color','k','LineStyle',':');
%                 
%                 axis tight
%                 legend('true','fake','Location','Best');legend boxoff;
%                 xlabel('time (s)');
%                 title(sprintf('xcorr neuron %g & %g',i,j));
%                 
%             end
%             
%             
%         end
        
        T = cell2mat({data.T});
        figure;clf;set(gcf,'color','w');hold on;
        
        %conds is only for numel of real trials, so not averaging over more
        %tirals
        blah1 = 0; blah2 = 0;
        for ii = 1:n_conds
            temp1 = (nanmean(psth(i,:,conds==ii),3) - nanmean(fake_psth(i,:,conds_fake==ii),3)).^2;
            blah1 = blah1 + sum(temp1(~isnan(temp1)));
            temp2 = (nanmean(fake_psth(i,:,conds==ii),3)).^2;
            blah2 = blah2 + sum(temp2(~isnan(temp2)));
        end
        
        for ii = 1:n_conds
            
            plot(0:dt:(ceil(max(T/dt))-1)*dt,nanmean(psth(i,:,conds==ii),3),'color',clrs(ii,:),...
                'LineWidth',1,'Linestyle',':');
            plot(0:dt:(ceil(max(T/dt))-1)*dt,nanmean(fake_psth(i,:,conds_fake==ii),3),'color',clrs(ii,:),...
                'LineWidth',1,'Linestyle','-');
            
            %errorbar(0:nplot*dt:(ceil(max(T/dt))-1)*dt,nanmean(psth(i,1:nplot:end,conds==ii),3),...
            %    nanstderr(psth(i,1:nplot:end,conds==ii),3),...
            %    'color',clrs(ii,:),'LineWidth',1,'Linestyle','-');
            %errorbar(0:nplot*dt:(ceil(max(T/dt))-1)*dt,nanmean(fake_psth(i,1:nplot:end,conds==ii),3),...
            %    nanstderr(fake_psth(i,1:nplot:end,conds==ii),3),...
            %    'color',clrs(ii,:),'LineWidth',2,'Linestyle',':');
            
            %plot mean of real
            X1 = nanmean(psth(i,:,conds==ii),3) - nanstderr(psth(i,:,conds==ii),3);
            X1 = X1(~isnan(X1));
            X2 = fliplr(nanmean(psth(i,:,conds==ii),3) + nanstderr(psth(i,:,conds==ii),3));
            X2 = X2(~isnan(X2));
            
            h = fill([(0:numel(X1)-1)*dt,fliplr((0:numel(X2)-1)*dt)],...
               [X1, X2],clrs(ii,:));           
            set(h,'facealpha',.025);
            
%             X1 = nanmean(fake_psth(i,:,conds==ii),3) - nanstderr(fake_psth(i,:,conds==ii),3);
%             X1 = X1(~isnan(X1));
%             X2 = fliplr(nanmean(fake_psth(i,:,conds==ii),3) + nanstderr(fake_psth(i,:,conds==ii),3));
%             X2 = X2(~isnan(X2));
%             
%             h2 = fill([(0:numel(X1)-1)*dt,fliplr((0:numel(X2)-1)*dt)],...
%                 [X1, X2],clrs(ii,:));
%             set(h2,'facealpha',0.05);
    
        end
        
        legend('data','stimulated data','Location','Best');legend boxoff;
        set(gca,'xlim',[0 0.5]);
        yl = get(gca,'ylim');
        set(gca,'ylim',[0 yl(2)]);
        xlabel('Time from stimulus onset (s)');ylabel('Firing rate (Hz)');
        title(sprintf('PSTH Neuron %g, R2: %g',i, 1 - blah1/blah2));
        
    end
end

% for i = 1:numel(data)
%     if data(i).nT > 20
%         figure(1001);clf;set(gcf,'color','w','Menubar','none');
%         subplot(5,1,[3:5]);imagesc([0:data(i).nT-1]*dt,xc,alpha{i});
%         xlabel('time (s)');ylabel('accumulator');
%         set(gca,'YDir','normal');
%         subplot(5,1,[1:2]);plot([0:data(i).nT-1]*dt,...
%             FilterSpikes(2,data(i).spike_counts(:,3)/dt),'LineWidth',2);
%         set(gca,'ylim',[0 70],'XColor','w','box','off','LineWidth',2);
%         ylabel('firing rate (Hz)');
%         title(sprintf('trial %g',i));
%         drawnow;
%         pause(5);
%     end
% end
