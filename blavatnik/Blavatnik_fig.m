clearvars;

in_pth = '~/Documents/Dropbox/hanks_data_sessions';
ratname = 'T036';

if strcmp(ratname,'T035')
    sessid = {'169448','167725','166135','164900'};
    %sessid = {'166135','166590','169448','163098','163885','164449',...
    %    '164752','164900','165058','167725','167855','168628'};
elseif strcmp(ratname,'T036')
    sessid = {'157201','157357','157507','168499'};
    %sessid = {'154154','154291','154448','154991','155124','155247','155840',...
    %    '157201','157357','157507','168499','168627'};
    %sessid = {'157201'};
end

%save_pth = fullfile('~/Documents/Dropbox/results/multiple_session/',[ratname,'_6_4_param_1e_32']);
save_pth = fullfile('~/Documents/Dropbox/results/multiple_session/',[ratname,'_4sess__6_4_param_1e_32']);

model_type = 'spikes';

%load parameters from fit with multiple session
load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid),model_type)));

%create vector 
%xgen = x0;
%xgen(fit_vec) = xf;

%load data used to fit the parameters
%[data] = load_data(ratname,model_type,sessid,in_pth);

sessid = {'157201'};usen = 3; usen2 = 1; %FOF 5353
%sessid = {'169448'};usen = 2; %PPC 6217

%load data for which this cell has data
[data] = load_data(ratname,model_type,sessid,in_pth);

use2.choice = false; use2.spikes = false;
forward_data = data;
for i = 1:numel(forward_data)
    forward_data(i).N = [];
end
%generate from the forward model
[~,xc,alpha] = LL_all_trials(xf,forward_data,dt,n,nstdc,dimz,dimd,dimy,...
    use2,0,settle,fr_func,true,x0(~fit_vec),fit_vec);

%load rawdata to check various things, like if it's the neurons I think it
%should be
%load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{1})),'rawdata');
%[~,Asamp] = sample_model_v4(rawdata,xgen,use);

tri = numel(data);

% make clrs cell array if it isn't provided
n_conds = 8;

max_br = 1;
min_br = 0.4;
n_each_clr = floor(n_conds/2);
br_vec = linspace(min_br,max_br,n_each_clr);
for i=1:n_conds
    if i <= n_each_clr
        % reds for low conditions
        clrs(i,:) = [br_vec(i) 0 0];
    elseif i > ((n_conds+1)/2)
        % greens for high conditions
        clrs(i,:) = [0 br_vec(end-i+n_conds-n_each_clr+1)  0];
    else
        % black for middle condition is there is one
        clrs(i,:) = [0.7 0.7 0];
    end
end
clrs = flipud(clrs);

T = cell2mat({data.T});
psth = NaN(ceil(max(T/dt)),tri);
A = NaN(ceil(max(T/dt)),tri);
psth2 = NaN(ceil(max(T/dt)),tri);

for j = 1:tri
    if data(j).nT > 3
        psth(1:size(data(j).spike_counts(:,usen),1),j) = FilterSpikes(2,data(j).spike_counts(:,usen)/dt)';
        psth2(1:size(data(j).spike_counts(:,usen2),1),j) = FilterSpikes(2,data(j).spike_counts(:,usen2)/dt)';
        A(1:size(data(j).spike_counts(:,usen),1),j) = FilterSpikes(2,alpha{j}'*xc')';
        %A(:,j) = decimate(Asamp(j,:),200);
        %temp = nanmean(reshape(Asamp(j,1:floor(size(Asamp(j,:),2)/200)*200),200,[]),1);
        %A(1:numel(temp),j) = temp;
    end
end

AA = NaN(ceil(max(T/dt)),n_conds);
PP = NaN(ceil(max(T/dt)),n_conds);
PP2 = NaN(ceil(max(T/dt)),n_conds);
%Conds = NaN(size(A));

for i = 1:ceil(max(T/dt))-1
    %conds = bin_vals(A(i,:), n_conds, 'split_zero', 1);
    %Conds(i,:) = conds;
    temp = A(i,:);
    conds = discretize(temp,linspace(min(temp)-eps,max(temp)+eps,n_conds+1));
    for ii = 1:n_conds
        AA(i,ii) = nanmean(A(i,conds==ii),2);
        PP(i,ii) = nanmean(psth(i,conds==ii),2);
        PP2(i,ii) = nanmean(psth2(i,conds==ii),2);
    end
end

%%

figure;set(gcf,'color','w');

for ii = 1:n_conds
    subplot(3,1,1);hold on;
    plot(0:dt:(ceil(max(T/dt))-1)*dt,AA(:,ii),...
        'color',clrs(ii,:),'LineWidth',1);
    xlabel('time (s)');
    %ylabel(sprintf('Decision \n variable'));
    title('Decision Variable','FontWeight','normal');
    set(gca,'xlim',[0 0.5],'XColor','w');
    
    subplot(3,1,2);hold on;
    plot(0:dt:(ceil(max(T/dt))-1)*dt,PP(:,ii),...
        'color',clrs(ii,:),'LineWidth',1);
    %xlabel('Time from stimulus onset (s)');
    ylabel(sprintf('Firing \n rate (Hz)'));
    title('Neuron 1','FontWeight','normal');
    set(gca,'xlim',[0 0.5],'XTick',linspace(0,0.5,6),...
        'ylim',[10 50],'YTick',[10:10:40],...
        'XColor','w');
    
    subplot(3,1,3);hold on;
    plot(0:dt:(ceil(max(T/dt))-1)*dt,PP2(:,ii),...
        'color',clrs(ii,:),'LineWidth',1);
    xlabel('Time from stimulus onset (s)');ylabel(sprintf('Firing \n rate (Hz)'));
    title('Neuron 2','FontWeight','normal');
    set(gca,'xlim',[0 0.5],'XTick',linspace(0,0.5,6));
    
    
end


