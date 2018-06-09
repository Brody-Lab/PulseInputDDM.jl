function [x0,fit_vec] = initalize_v3(data,N,model_type)

% Set initial parameters values and lower and uppper bounds for
% optimization

%create some new global variables which will be used by many functions
global dimy

%% parameters for latent variable

%          vari       inatt          B    lambda      vara    vars     phi    tau_phi 
fit_vec = [false(1,1);false(1,1);    true(4,1);                         false(2,1)];

%these will be the defaults it fit_vec is false for an entry
%if any(strcmp(model_type,{'spikes','joint'}))
    x0 =  [1e-5       0.0,           20,    1e-3,     10,     0.99,      0.99,  0.2]';   
%else
%    x0 =  [1e-5       0.0,           10,   0.0,      10,     1.0,      1.0,  0.02]';
%end

%% Data likelihood parameters

%if using choice data in the optimization
if strcmp(model_type,'choice') || strcmp(model_type,'joint');
    
    x0 = cat(1,x0,0);
    fit_vec = cat(1,fit_vec,true);
    
end

%....if using neural data
if strcmp(model_type,'spikes') || strcmp(model_type,'joint');
    
    %make an initial guess of the neural tuning curve parameters using the
    %stimulus data
    y0 = init_x0y_v2(data);
    
    %concatenate all values onto the end of existing vectors
    x0 = cat(1,x0,reshape(y0',[],1));
    fit_vec = cat(1,fit_vec,true(dimy*N,1));
    
end
