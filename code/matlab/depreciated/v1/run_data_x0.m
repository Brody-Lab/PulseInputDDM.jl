function [xf,history,x0,fit_vec,data] = ...
    run_data_x0(ratname,model_type,sessid,in_pth,save_pth)

%for generating x0 and lb and data for doing optimization in julia

global dt N use

sessid = strsplit(sessid,'_'); %turn string of sessions, where each session is separate by a _, into a cell array

[data,use,N] = load_data(ratname,model_type,sessid,in_pth,dt);         %load and "package" the data
save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');         %save "packaged" data
fprintf('saving "packaged" data\n');

preamble;         %set various global parameters
[x0,lb,ub,fit_vec] = initalize(data,use,N);     %bounds and inital point
history = struct('x',[]);         %preallocate to save optimization history
if ~exist(save_pth); mkdir(save_pth); end         %make new direcdtory if it does not exist

%save settings
save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
    'history',...
    'x0','lb','ub','fit_vec',...
    'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle',...
    'use','N');
fprintf('save initialized parameters\n');

end
