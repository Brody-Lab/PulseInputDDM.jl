function [xf,history,x0,fit_vec,data] = ...
    run_data_x0_w_gen(ratname,model_type,sessid,in_pth,save_pth,gen)

%for generating x0 and lb and data for doing optimization in julia

global dt n nstdc dimz dimd settle dimy fr_func N use

sessid = strsplit(sessid,'_'); %turn string of sessions, where each session is separate by a _, into a cell array

try
    
    %set location of parfor directory
    if strcmp(getenv('SLURM_CLUSTER_NAME'),'della')
        pc = parcluster('local');
        pc.JobStorageLocation = fullfile('/tigress/',getenv('USER'),...
            'latent_accum_models/junk',getenv('SLURM_ARRAY_TASK_ID'));
        parpool(pc,pc.NumWorkers);
    end
   
    fprintf('no saved parameters or data\n');
    %set various global parameters
    preamble;
    %load and process the data
    fprintf('constructing real data, given rat and sessions\n');
    [data,use,N] = load_data(ratname,model_type,sessid,in_pth);
    %bounds and inital point
    [x0,lb,ub,fit_vec] = initalize(data,use,N);
    
    %preallocate to save optimization history
    history = struct('x',[]);
    
    %make new direcdtory if it does not exist
    if ~exist(save_pth)
        mkdir(save_pth)
    end
    
    %save fmincon output
    save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
        'history',...
        'x0','lb','ub','fit_vec',...
        'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle',...
        'use','N');
    fprintf('save initialized parameters\n');
    
    %generate fake data
    if gen
        
        fprintf('starting fit, running in generative mode\n');
        fprintf('generating fake data\n');
        %generate fake data for testing model
        [data,xgen] = generate_data(ratname,use,sessid,in_pth);
        
        %for plotting psth of data
        %[cta,psth,conds] = psth_cta(data);
        %plot_psth(conds,psth,data);
        
        %bounds and inital point
        fprintf('computing x0 based on fake data\n');
        %x0 = initalize(data,use,N,xgen); %initalize at generative parameters
        x0 = initalize(data,use,N);
        %save fake data
        save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');
        LL_star = LL_all_trials(xgen(fit_vec),data,dt,n,nstdc,dimz,dimd,dimy,...
            use,N,settle,fr_func,[],xgen(~fit_vec),fit_vec);
        fprintf('LL_star: %g\n',LL_star);
        fprintf('saving fake data\n');
        
        %save fmincon output
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'x0','xgen','-append');
        fprintf('overwritting x0 (based on fake data) and saving xgen\n');
        
    end
   
    xf = x0(fit_vec);
    
    %save fmincon output
    save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
        'xf','-append');
    
    delete(gcp);
    
catch me
    
    fprintf('%s / %s\n',me.identifier,me.message);
    delete(gcp);
    
end

end
