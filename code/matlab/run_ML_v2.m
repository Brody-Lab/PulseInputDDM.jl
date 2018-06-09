function [xf,history,x0,fit_vec,data] = run_ML_v2(ratname,model_type,sessid,in_pth,save_pth,reload)

global dt N n dimy dimz dimd fr_func

if nargin < 6 || isempty(reload)
    reload = 'none';
end

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
    
    %% load or create global parameters and settings
    
    try
        %check for and load ML parameters and settings from a previous fit
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'dt','n','dimz','dimd','dimy','fr_func');
        fprintf('reloading existing global parameters\n');
        
    catch
        preamble_v2;         %set various global parameters
        %save settings
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'dt','n','dimz','dimd','dimy','fr_func');
        fprintf('save initialized global parameters\n');
        
    end
    
    %% load or create "packaged" data
    try
        %load previously packaged data, located in the same directory
        %as the ML parameters
        load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'N');
        fprintf('reloading existing data\n');
        
    catch
        [data,N] = load_data_v2(ratname,sessid,in_pth,dt,model_type);         %load and "package" the data
        save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');         %save "packaged" data
        %save properties of the data
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'N','-append');
        fprintf('saving "packaged" data\n');
        
    end
    
    %% load x0 and fit_vec (which parameters to fit)
    
    mfile = matfile(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
    %load or create bound parameters and initialization
    if any(strcmp(fieldnames(mfile),'fit_vec'))
        %check for and load ML parameters and settings from a previous fit
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'x0','fit_vec');
        fprintf('reloading existing bound parameters\n');
        
    else
        [x0,fit_vec] = initalize_v3(data,N,model_type);     %bounds and inital point
        %save bounds
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'x0','fit_vec','-append');
        fprintf('save bound parameters\n');
        
    end
    
    %% Reload a previous solution
    
    switch reload
        
        case 'julia'
            
            try
                load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
                    'xf');
                
                %use previous final ML solution as intial solution
                x0(fit_vec) = xf;
                fprintf('julia final\n');
                
            catch
                
                history_j = load(fullfile(save_pth,sprintf('julia_history_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
                    'history');
                %use previous final ML solution as intial solution
                x0(fit_vec) = history_j(:,end);
                fprintf('julia last from history\n');
                
                if any(strcmp(fieldnames(mfile),'history'))
                    %check for and load ML parameters and settings from a previous fit
                    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
                        'history');
                    history.x = cat(2,history.x,history_j);
                end
                
            end
            
        case 'matlab'
            
            %check for and load ML parameters and settings from a previous fit
            load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
                'history','xf');
            
            if exist('xf')
                %use previous final ML solution as intial solution
                x0(fit_vec) = xf;
                fprintf('reloading existing ML parameters from xf\n');
            else
                %use previous final ML solution as intial solution
                x0(fit_vec) = history.x(:,end);
                fprintf('reloading existing ML parameters from matlab history\n');
            end
            
        otherwise
            
            history = struct('x',[]);         %preallocate to save optimization history
            if ~exist(save_pth); mkdir(save_pth); end         %make new direcdtory if it does not exist
            %save settings
            save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
                'history','-append');
            fprintf('starting at x0 and saved history\n');
            
    end
      
    %% Run fminunc
    
    %optimization options
    options = optimoptions(@fminunc,'OutputFcn',...
        @(x,optimValues,state,timelimit)outfun(x,optimValues,state),...
        'Display','iter-detailed','MaxIter',inf,...
        'Algorithm', 'quasi-newton','OptimalityTolerance',1e-6,...
        'MaxFunctionEvaluations',inf,'StepTolerance',1e-128,'UseParallel',true,...
        'FunctionTolerance',1e-12);
    
    x0 = deparameterize(x0,model_type,N);
    
    %optimization objective function
    fobj = @(x) ll_wrapper(x,x0,fit_vec,data,model_type,dt,n,dimz,dimd,dimy,N,fr_func);
    
    tic; xf = fminunc(fobj,x0(fit_vec),options); toc;     %run optimization
    
    xf = reparameterize(x0,dt,model_type,N,xf,fit_vec);
    xf = xf(fit_vec);
    %save fmincon output
    save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
        'xf','history','-append');
    
    delete(gcp);
    
catch me
    
    fprintf('%s / %s\n',me.identifier,me.message);
    delete(gcp);
    
end

%For printing and saving history between iterations
    function stop = outfun(x,~,state)
        time = toc;
        
        switch state
            case 'iter'
                xtemp = reparameterize(x0,dt,model_type,N,x,fit_vec);
                xtemp = xtemp(fit_vec);
                history.x = cat(2,history.x,xtemp);
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
