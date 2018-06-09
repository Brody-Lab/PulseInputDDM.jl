function [xf,history,x0,fit_vec,data] = run_ML_w_gen(ratname,model_type,sessid,in_pth,save_pth,gen)

global dt n nstdc dimz dimd settle dimy fr_func N use

sessid = strsplit(sessid,'_'); %turn string of sessions, where each session is separate by a _, into a cell array
timelimit = 23 * 3600; tic;

try
    
    %set location of parfor directory
    if strcmp(getenv('SLURM_CLUSTER_NAME'),'della') || strcmp(getenv('SLURM_CLUSTER_NAME'),'spock')
        pc = parcluster('local');
        pc.JobStorageLocation = fullfile(getenv('HOME'),'junk', getenv('SLURM_ARRAY_JOB_ID'),...
            getenv('SLURM_ARRAY_TASK_ID'));
        parpool(pc,pc.NumWorkers);
    end
    
    %check for existing initial points
    if exist(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)))
        
        %load it if it exists
        load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'history',...
            'x0','lb','ub','fit_vec',...
            'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle',...
            'use','N');
        fprintf('reloading existing parameters\n');
        
        %load data that is saved in same directory
        if exist(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)))
            
            load(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data'); 
            fprintf('reloading existing data\n');

            %if running in generative, reload generative parameters
            if gen
                load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xgen');
                fprintf('reloading existing generative parameters\n');
                LL_star = LL_all_trials(xgen(fit_vec),data,dt,n,nstdc,dimz,dimd,dimy,...
                    use,N,settle,fr_func,[],xgen(~fit_vec),fit_vec);
                fprintf('LL_star: %g\n',LL_star);               
            end   
            
        else
            
            %if no data, and want to generate it
            if gen
                
                %reload generative parameters
                load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xgen');
                fprintf('reloading existing generative parameters\n');
                
                %2/14 added dt input now, since I will also want to use
                %this code post-hoc for generating data with particularly
                %binned spikes
                data = generate_data(ratname,use,sessid,in_pth,xgen,[],[],dt);
                fprintf('generating data from generative parameters\n');

                %bounds and inital point
                x0 = initalize(data,use,N);
                fprintf('computing x0\n');

                %save fake data
                save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');
                LL_star = LL_all_trials(xgen(fit_vec),data,dt,n,nstdc,dimz,dimd,dimy,...
                    use,N,settle,fr_func,[],xgen(~fit_vec),fit_vec);
                fprintf('LL_star: %g\n',LL_star);
                fprintf('saving generated data\n');

                %append x0
                save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'x0','-append');   
                fprintf('saving new x0 based on generated data data\n');
                
            else
                
                data = load_data(ratname,model_type,sessid,in_pth,dt);
                fprintf('constructing real data, given rat and sessions\n');
                save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');
                fprintf('saving constructed real data\n');

            end
        end
        
        try
            %...and use previous final solution as intial solution
            x0(fit_vec) = history.x(:,end);
        catch
            history = struct('x',[]);
        end
        
    else          
        
        fprintf('no saved parameters or data\n');
        %set various global parameters
        preamble;
        %load and process the data
        fprintf('constructing real data, given rat and sessions\n');
        [data,use,N] = load_data(ratname,model_type,sessid,in_pth,dt);
        %bounds and inital point
        [x0,lb,ub,fit_vec] = initalize(data,use,N);
                      
        %preallocate to save optimization history
        history = struct('x',[]);
        
        %make new direcdtory if it does not exist
        if ~exist(save_pth); mkdir(save_pth); end
        
        %save fmincon output
        save(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),...
            'history',...
            'x0','lb','ub','fit_vec',...
            'dt','n','nstdc','dimz','dimd','dimy','fr_func','settle',...
            'use','N');
        fprintf('save initialized parameters\n');
        
        save(fullfile(save_pth,sprintf('data_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'data');
        fprintf('saving constructed real data\n');
    end
    
    %optimization options
    options = optimoptions(@fmincon,'OutputFcn',...
        @(x,optimValues,state,timelimit)outfun(x,optimValues,state),...
        'Display','iter-detailed','MaxIter',inf,...
        'Algorithm', 'interior-point','OptimalityTolerance',1e-6,...
        'MaxFunctionEvaluations',inf,'StepTolerance',1e-32,'UseParallel',true,...
        'FunctionTolerance',1e-32);
    
    %optimization objective function
    fobj = @(x) LL_all_trials(x,data,dt,n,nstdc,dimz,dimd,dimy,...
        use,N,settle,fr_func,[],x0(~fit_vec),fit_vec);
        
    %run optimization
    tic; xf = fmincon(fobj,x0(fit_vec),[],[],[],[],lb,ub,[],options); toc;
    
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
                %save current value of histroy
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
