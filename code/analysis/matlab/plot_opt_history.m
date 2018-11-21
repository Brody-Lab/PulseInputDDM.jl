function xf = plot_opt_history(ratname,sessid,model_type,save_pth)

global dimz

sessid = strsplit(sessid,'_'); %turn string of sessions, where each
load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
if ~exist('xf')
    xf = history.x(:,end);
end

if 1
if exist(fullfile(save_pth,sprintf('julia_history_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)))
    load(fullfile(save_pth,sprintf('julia_history_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    blah = history;
    clear history;
    history.x = blah;
end
end

%for julia
if 0
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    blah = cell2mat(history);
    history = [];
    history.x = reshape(blah,8,[]);
    CI = 2*sqrt(diag(inv(H)));
end

%for bing
if 0
    history.x = reshape(history.x,sum(fit_vec),[]);
    history.x = history.x([8,4,1,2,3,5,6,7],:);
    history.x(5,:) = history.x(5,:)/40;
end

try
    load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    CI = 2*sqrt(diag(inv(H)));
end

figure;clf;set(gcf,'color','w');
latent_str = {'\sigma_i','inatt','B','\lambda','\sigma_a','\sigma_s','\phi','\tau_\phi'};

vec = find(fit_vec(1:dimz));

for i = 1:numel(vec)
    
    subplot(2,ceil(numel(vec)/2),i); hold on;
    plot(history.x(i,:)','LineWidth',2);
    
    if exist('H')
        errorbar(numel(history.x(i,:)),history.x(i,end),CI(i),...
            'LineWidth',2);
    end
    
    if exist('xgen')
        plot(linspace(0,numel(history.x(i,:)),100),xgen(vec(i)) * ones(100,1),...
            'g--','LineWidth',2);
    end
    
    title(latent_str{vec(i)}); xlabel('iter'); set(gca,'box','off');
    %set(gca,'ylim',[min([0.9 * min(history.x(i,:)),0.9 * min(xgen(vec(i)))]),...
    %    max([1.1 * max(history.x(i,:)),1.1 * max(xgen(vec(i)))])]);
    
end
