clearvars;

ratname = 'T036';

%in_pth = '~/Dropbox/hanks_choices'; model_type = 'choice'; sessid = '1';
%fake data, choices, 5e4 trials
%out_pth = fullfile('~/Dropbox/results/choices/fake_data/',ratname,sessid,'5e4trials');
%fake data, choices, orig # of trials
%out_pth = fullfile('~/Dropbox/results/choices/fake_data/',ratname,sessid,'me_nosettle_rx0');

in_pth = '~/Documents/Dropbox/hanks_data_sessions'; model_type = 'spikes';

sessid = '157201_157357_157507_168499';

%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new');
out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'new_lowerlimits');
%out_pth = fullfile('~/Dropbox/results/multiple_session/',sessid,'old');
%fake data, original number of trials
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'orig_trials');
%fake data, more trials
%out_pth = fullfile('~/Dropbox/results/multiple_session/fake_data/',ratname,sessid,'15e3trials');

plot_opt_history(ratname,sessid,model_type,out_pth);

