function [xf,history,x0,fit_vec,data] = run_ML_unc(ratname,model_type,sessid,in_pth,save_pth)

global dt n nstdc dimz dimd settle dimy fr_func N use

sessid = strsplit(sessid,'_'); %turn string of sessions, where each session is separate by a _, into a cell array
timelimit = 23 * 3600; tic;

try
    
    %set location of parfor directory when running on cluster
    if strcmp(getenv('SLURM_CLUSTER_NAME'),'della') || ...
            strcmp(getenv('SLURM_CLUSTER_NAME'),'spock')
        pc = parcluster('local');
        pc.JobStorageLocation = fullfile(getenv('HOME'),'junk', ...
            getenv('SLURM_ARRAY_JOB_ID'),getenv('SLURM_ARRAY_TASK_ID'));
        parpool(pc,pc.NumWorkers);
    end
    
    %load or create global parameters and settings
    try        
        %check for and load ML parameters and settings from a previous fit
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle');        
        fprintf('reloading existing global parameters\n');
        
    catch               
        preamble;         %set various global parameters        
        %save settings
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle');
        fprintf('save initialized global parameters\n');
        
    end
    
    %load or create "packaged" data
    try
        %load previously packaged data, located in the same directory
        %as the ML parameters
        load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data'); 
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'use','N');
        fprintf('reloading existing data\n');
        
    catch
        [data,use,N] = load_data(ratname,model_type,sessid,in_pth,dt);         %load and "package" the data
        save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');         %save "packaged" data
        %save properties of the data
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'use','N','-append');
        fprintf('saving "packaged" data\n');
        
    end
    
    %look at which fields are located in the bundled .mat file from
    %whatever was already run
    mfile = matfile(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
    %load or create bound parameters and initialization
    if any(strcmp(fieldnames(mfile),'lb'))     
        %check for and load ML parameters and settings from a previous fit
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'x0','lb','ub','fit_vec');     
        fprintf('reloading existing bound parameters\n');
        
    else               
        [x0,lb,ub,fit_vec] = initalize_unc(data,use,N);     %bounds and inital point                         
        %save bounds
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'x0','lb','ub','fit_vec','-append');
        fprintf('save bound parameters\n');
        
    end
    
    %load or create ML parameters and settings
    if any(strcmp(fieldnames(mfile),'history'))     
        %check for and load ML parameters and settings from a previous fit
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'history','xf','x0new');
        
        if exist('x0new')
            %use new x0 as x0
            x0(fit_vec) = x0new;
        elseif exist('xf')
            %use previous final ML solution as intial solution
            x0(fit_vec) = xf;
        else   
            try
                %use previous final ML solution as intial solution
                x0(fit_vec) = history.x(:,end);
            end
        end
        
        fprintf('reloading existing ML parameters\n');
        
    else               
        history = struct('x',[]);         %preallocate to save optimization history
        if ~exist(save_pth); mkdir(save_pth); end         %make new direcdtory if it does not exist      
        %save settings
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'history','-append');
        fprintf('save initialized parameters\n');
        
    end
    
%%
    
    %optimization options
    options = optimoptions(@fminunc,'OutputFcn',...
        @(x,optimValues,state,timelimit)outfun(x,optimValues,state),...
        'Algorithm', 'quasi-newton','Display','iter-detailed','MaxIter',inf,...
        'OptimalityTolerance',1e-6,...
        'MaxFunctionEvaluations',inf,'StepTolerance',1e-32,'UseParallel',true,...
        'FunctionTolerance',1e-32);
    
    %optimization objective function
    fobj = @(x) LL_all_trials_unc(x,data,dt,n,nstdc,dimz,dimd,dimy,...
        use,N,settle,fr_func,[],x0(~fit_vec),fit_vec,lb,ub);
        
    %tic; xf = fmincon(fobj,x0(fit_vec),[],[],[],[],lb,ub,[],options); toc;     %run optimization
     tic; xf = fminunc(fobj,x0(fit_vec),options); toc;     %run optimization
    
    %save fmincon output
    save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
        'xf','history','-append'); 
    
    delete(gcp);
    
catch me
    
    fprintf('%s / %s\n',me.identifier,me.message);
    delete(gcp);
    
end

    function stop = outfun(x,~,state)               
        time = toc;
        
        switch state           
            case 'iter'               
                history.x = cat(2,history.x,x);            
                %save current value of history
                save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'history','-append');        
        end
        
        if time < timelimit           
            stop = false;
            
        else
            stop = true;
            fprintf('hit time limit!\n');
            
        end
        
    end

end
