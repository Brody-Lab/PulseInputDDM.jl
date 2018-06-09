function make_data(ratname,model_type,sessid,in_pth,save_pth)

global dt N n dimy dimz dimd fr_func

sessid = strsplit(sessid,'_'); %turn string of sessions, where each session is separate by a _, into a cell array
preamble_v2;         %set various global parameters
[data,N] = load_data_v2(ratname,sessid,in_pth,dt,model_type);         %load and "package" the data
save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data'); 
x0 = initalize_v3(data,N,model_type);     %bounds and inital point
save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'x0');

end
