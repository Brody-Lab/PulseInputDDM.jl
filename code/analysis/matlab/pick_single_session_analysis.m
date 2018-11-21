function pick_single_session_analysis

in_pth = '~/Documents/Dropbox/hanks_data_sessions';

%save_pth = '~/Documents/Dropbox/results/T036_6_4_param_1e_32';
%ratname = 'T036';
% sessids = {'154154','154291','154448','154991','155124','155247','155840',...
%     '157201','157357','157507','168499','168627'};
%sessids = {'157201'};

%169448 has cell 6217
save_pth = '~/Documents/Dropbox/results/T035_6_4_param_1e_32';
ratname = 'T035';
sessids = {'166135','166590','169448','163098','163885','164449',...
    '164752','164900','165058','167725','167855','168628'};

for i = 1:numel(sessids)
    
    clc;
    sessid = sessids{i};
    x(i,:) = post_fit_analysis(ratname,'spikes',{sessid},in_pth,save_pth);
    fprintf('rat: %s, session %s\n',ratname,sessid);
    keyboard; close all
    
end

latent_str = {'\sigma_i','inatt','B','\lambda','\sigma_a','\sigma_s','\phi','\tau_\phi'};
figure;
for i = 1:size(x,2)
    subplot(2,4,i)
    plot(x(:,i),'x');
    title(latent_str{vec(i)});    
end