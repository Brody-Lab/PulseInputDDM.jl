
clearvars;

%%

%cell 4 =  5381; cell 7 = 5388;

in_pth = '~/Projects/inProgress/spike-data_latent-accum/data/hanks_data_sessions/';
model_type = 'spikes';
sessid = '157201_157357_157507_168499'; ratname = 'T036';
%sessid = '169448_167725_166135_164900'; ratname='T035';
%sessid = '195676_196336_196580_196708_198004'; ratname = 'T063';
%sessid = '153219_153382_153510_154806_154950_155375_155816_155954_157026_157178_157483_158231_161057_161351_164574_165972'; ratname = 'T011';
%sessid = '297609_298403_300465_300634_301314_301749_302840';ratname = 'T080';
%sessid = '304258_304450';ratname = 'T103';

%     in_pth = '~/Dropbox/hanks_choices'; model_type = 'choice'; sessid = '1';
%     sessid = '169448_167725_166135_164900';
%     sessid = '166135_166590_169448_163098_163885_164449_164752_164900_165058_167725_167855_168628';
%     sessid = '154154_154291_154448_154991_155124_155247_155840_157201_157357_157507_168499_168627';

%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'old');
%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new_lowerlimits');
out_pth = fullfile('~/Projects/inProgress/spike-data_latent-accum/data/results/julia/',model_type,sessid,'17855745');
%out_pth = fullfile('~/Dropbox/results/archive/settling');
%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'2_param_rate_func');
%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new');
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'orig_trials');
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'15e3trials');

%glm_choice(ratname,model_type,sessid,in_pth,out_pth)
%[LL,dd,V,H,params] = LL_land_analysis_v2(ratname,model_type,sessid,out_pth,true);
%[XS,LL] = LL_land_analysis_grad(ratname,model_type,sessid,out_pth);
post_fit_analysis(ratname,model_type,sessid,in_pth,out_pth)
%[xf,xy] = post_fit_psths(ratname,model_type,sessid,in_pth,out_pth);
%[xf,xy,mtcurve,CI,R2] = post_fit_params_and_good_fit(ratname,model_type,sessid,in_pth,out_pth);

%%

figure;subplot(1,5,1:5);imagesc(1:52,52-5:52,V(1:end,end-5:end)');set(gca,'XTick',1:62,'XTickLabel',params([3,4,5,6,9:end]));
caxis([-0.6 0.6])
colorbar

%%

figure;
vec = 48:52;
for i = 1:5
    subplot(5,1,i);
    plot(1:48,V(5:end,vec(i)),'x')
    set(gca,'XTick',1:48,'XTickLabel',params(9:end));
    title(sprintf('EV %g',vec(i)));
end

%%

in_pth = '~/Dropbox/hanks_data_sessions';
model_type = 'spikes';
ratname = {'T036','T035','T063','T011','T080','T103'};
sessid = {'157201_157357_157507_168499',...
    '169448_167725_166135_164900',...
    '195676_196336_196580_196708_198004',...
    '153219_153382_153510_154806_154950_155375_155816_155954_157026_157178_157483_158231_161057_161351_164574_165972',...
    '297609_298403_300465_300634_301314_301749_302840',...
    '304258_304450'};

for i = 1:numel(ratname)
    
    out_pth = fullfile('~/Dropbox/results/new/',sessid{i});    
    [xf{i},xy{i},mtcurve{i},CI{i},R2{i}] = post_fit_params_and_good_fit(ratname{i},model_type,sessid{i},in_pth,out_pth);
end

%% parameters

figure(1);clf;set(gcf,'color','w','Menubar','none');

latent_str = {'B','\lambda','\sigma_a','\sigma_s','\phi','\tau_\phi'};
color = {'r','b','r','b','g','g'};

for j = 1:6
    
     subplot(2,3,j);hold on;
    
    for i = 1:numel(xf)       
        
        errorbar(i,xf{i}(j),CI{i}(j),'Marker','s','LineStyle','none','LineWidth',1.5,'color',color{i},...
            'MarkerEdgeColor',color{i},'MarkerFaceColor',color{i},'MarkerSize',4);
        
    end
    
    title(latent_str{j});
    set(gca,'box','off','xlim',[0 7],...
        'XTick',linspace(1,6,6),'XTickLabel',ratname,...
        'XTickLabelRotation',90);
    
    
end

%% Tuning curve

figure(2);clf;

for i = 1:numel(mtcurve)
    
    subplot(2,3,i);hold on;
    
    X1 = nanmean(mtcurve{i}) - nanstderr(mtcurve{i});
    X1 = X1(~isnan(X1));
    X2 = fliplr(nanmean(mtcurve{i}) + nanstderr(mtcurve{i}));
    X2 = X2(~isnan(X2));
    
    h(i) = fill([linspace(-1,1,numel(X1)),fliplr(linspace(-1,1,numel(X1)))],...
        [X1, X2],color{i},'EdgeColor',color{i});
    set(h(i),'facealpha',.25);
    
    plot(linspace(-1,1,numel(X1)),mean(mtcurve{i}),'color',color{i},'LineWidth',2);
    title(ratname{i});

end

axis tight
set(gca,'linewidth',2,'fontsize',14,'YTick',linspace(0,1,3),...
    'ylim',[-0.05 1.05],'XTick',linspace(-1,1,3));
set(gca,'TickDir','out'); % The only other option is 'in'
xlabel({'Normalized accumulator value'},'fontweight','normal','FontSize',14);
ylabel({'Normalized firing rate'},'fontweight','normal','FontSize',14);

%%

clearvars;

in_pth = '~/Dropbox/hanks_data_sessions';
model_type = 'spikes';
sessid = '157201_157357_157507_168499'; ratname = 'T036';
out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new_lowerlimits');
[~,~,mtcurveFOF,~] = post_fit_analysis(ratname,model_type,sessid,in_pth,out_pth);

sessid = '169448_167725_166135_164900'; ratname='T035';
out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new_lowerlimits');
[~,~,mtcurvePPC,~] = post_fit_analysis(ratname,model_type,sessid,in_pth,out_pth);
%LL_land_analysis(ratname,model_type,sessid,in_pth,out_pth);

%% Just one neuron

ratname = 'T036';
model_type = 'spikes';
sessidstr = {'157201'};
in_pth = strcat('~/Dropbox/hanks_data_cells/',sessidstr{1});
sessid ='5353';
out_pth = fullfile('~/Dropbox/results/single_cells/',ratname,sessid);
post_fit_analysis(ratname,model_type,sessid,in_pth,out_pth);
LL_land_analysis(ratname,model_type,sessid,in_pth,out_pth);

%%

sessid = '157201_157357_157507_168499';
sessidstr = strsplit(sessid,'_');
count = 1;

for i = 1:numel(sessidstr)
    in_pth = strcat('~/Dropbox/hanks_data_cells/',sessidstr{i});
    files = dir(in_pth);
    for j = 1:numel(files)
        if ~isempty(strfind(files(j).name,'.mat'))
            sessid = strsplit(files(j).name,'.mat');
            sessid = sessid{1};
            sessid = sessid(regexp(sessid,'_') + 1: end);
            out_pth = fullfile('~/Dropbox/results/single_cells/',ratname,sessid);
            try
                %[xf(:,count),~,~,CI(:,count)] = post_fit_analysis(ratname,model_type,sessid,in_pth,out_pth);
                LL_land_analysis(ratname,model_type,sessid,in_pth,out_pth);
                count = count + 1;
            end
            keyboard;
        end
    end
end

%% Plot latent variable parameters

figure;clf;set(gcf,'color','w','Menubar','none');
fit_vec = [false(1,2),true(1,6)];
latent_str = {'\sigma_i','inatt','B','\lambda','\sigma_a','\sigma_s','\phi','\tau_\phi'};

for i = 1:sum(fit_vec)
    
    vec = find(fit_vec);
    subplot(2,ceil(sum(fit_vec)/2),i);
    errorbar(xf(i,:),CI(i,:),'Marker','s','LineStyle','none','LineWidth',1.5,'color',[255,99,71]/255,...
        'MarkerEdgeColor',[255,99,71]/255,'MarkerFaceColor',[255,99,71]/255,'MarkerSize',4);
    
    title(latent_str{vec(i)});
    xlabel('iter');
    set(gca,'box','off');
    
end
