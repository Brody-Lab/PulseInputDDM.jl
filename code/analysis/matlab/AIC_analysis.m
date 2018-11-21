clearvars;

%in_pth = '~/Documents/Dropbox/hanks_choices';
%ratname = 'T036';
%model_type = 'choice';
%sessid = '1';
%out_pth = fullfile('~/Documents/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'me_nosettle_rx0');

in_pth = '~/Documents/Dropbox/hanks_data_sessions';
ratname = 'T036';
model_type = 'spikes';
sessid = '157201_157357_157507_168499';
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'15e3trials');
out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new_lowerlimits');
%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'old');
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'orig_trials');

[LL,k] = compute_LL(ratname,model_type,sessid,in_pth,out_pth);

sessidstr = strsplit(sessid,'_');

%%

c = 1;
for i = 1:numel(sessidstr)
    in_pth = strcat('~/Dropbox/hanks_data_cells/',sessidstr{i});
    files = dir(in_pth);
    for j = 1:numel(files)
        if ~isempty(strfind(files(j).name,'.mat'))
            sessid = strsplit(files(j).name,'.mat');
            sessid = sessid{1};
            sessid = sessid(regexp(sessid,'_') + 1: end);
            out_pth = fullfile('~/Dropbox/results/single_cells/',ratname,sessid);
            [LLi(c),ki(c)] =  compute_LL(ratname,model_type,sessid,in_pth,out_pth);
            c = c + 1;
        end
    end
end

%%

AIC1 = 2 * k - 2 * LL;
AIC2 = 2 * sum(ki) - 2 * sum(LLi);

disp(abs(AIC1 - AIC2)/AIC1);
