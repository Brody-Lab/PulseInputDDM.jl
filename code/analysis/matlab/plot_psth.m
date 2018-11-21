function plot_psth(conds,psth,data)

global N dt

n_conds = numel(unique(conds));
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
for i = 1:N
    
    figure;clf;set(gcf,'color','w','Menubar','none');
    
    T = cell2mat({data.T});
    hold on;
    for ii = 1:n_conds
        plot(0:dt:(ceil(max(T/dt))-1)*dt,nanmean(psth(i,:,conds==ii),3),'color',clrs(ii,:),...
            'LineWidth',1,'Linestyle','-');
    end
    legend('true','fake','Location','Best');legend boxoff;
    set(gca,'xlim',[0 0.5]);
    xlabel('Time from stimulus onset (s)');ylabel('Firing rate (Hz)');
    
end

