clearvars;

load('/Users/briandepasquale/Documents/Dropbox/results/choice/7_1_param_1e_32/T036_1_choice.mat')
load('/Users/briandepasquale/Documents/Dropbox/results/choice/7_1_param_1e_32/H_T036_1_choice.mat')

blah(1,1) = xf(3);
temp = 2*sqrt(diag(inv(H)));
blah(1,2) = temp(3);

load('/Users/briandepasquale/Documents/Dropbox/results/choice/7_1_param_1e_32/T035_1_choice.mat')
load('/Users/briandepasquale/Documents/Dropbox/results/choice/7_1_param_1e_32/H_T035_1_choice.mat')

blah(2,1) = xf(3);
temp = 2*sqrt(diag(inv(H)));
blah(2,2) = temp(3);

load('/Users/briandepasquale/Documents/Dropbox/results/multiple_session/T036_4sess__6_4_param_1e_32/T036_157201 157357 157507 168499_spikes.mat')
load('/Users/briandepasquale/Documents/Dropbox/results/multiple_session/T036_4sess__6_4_param_1e_32/H_T036_157201 157357 157507 168499_spikes.mat');

blah(3,1) = xf(2);
temp = real(2*sqrt(diag(inv(H))));
blah(3,2) = temp(2);

load('/Users/briandepasquale/Documents/Dropbox/results/multiple_session/T035_4sess__6_4_param_1e_32/T035_169448 167725 166135 164900_spikes.mat')
load('/Users/briandepasquale/Documents/Dropbox/results/multiple_session/T035_4sess__6_4_param_1e_32/H_T035_169448 167725 166135 164900_spikes.mat')

blah(4,1) = xf(2);
temp = real(2*sqrt(diag(inv(H))));
blah(4,2) = temp(2);

color(1,:) = [255,99,71]/255;
color(2,:) = [64,224,208]/255;
color(3,:) = [255,250,205]/255;

figure;set(gcf,'color','w');hold on;
errorbar(1:2,blah(1:2,1),blah(1:2,2),'Marker','s','LineStyle','none','LineWidth',1.5,'color',[255,99,71]/255,'Capsize',16,...
    'MarkerEdgeColor',[255,99,71]/255,'MarkerFaceColor',[255,99,71]/255,'MarkerSize',4);
errorbar(3:4,blah(3:4,1),blah(3:4,2),'Marker','s','LineStyle','none','LineWidth',1.6,'color',[64,224,208]/255,'Capsize',16,...
    'MarkerEdgeColor',[64,224,208]/255,'MarkerFaceColor',[64,224,208]/255,'MarkerSize',4);
ylabel('$\lambda$','Interpreter','latex','Fontsize',18);
set(gca,'xlim',[0.75 4.1],'XTick',[1:4],'XTickLabel',{'T036-choice','T035-choice','T036-neural','T035-neural'},...
    'box','off','ylim',[-3 12],'YTick',[-2,0,3,6,9]);

%width of 4 height of 2

