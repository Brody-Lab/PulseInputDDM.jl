function [x0,lb,ub,fit_vec] = initalize_unc(data,use,N,xgen)

% Set initial parameters values and lower and uppper bounds for
% optimization

%create some new global variables which will be used by many functions
global dimy lby1 uby1

%variable input arguments
%provide a set of parameters, which will be used to initialize the
%parameters
if (nargin < 4 || isempty(xgen)); xgen = []; end

%% parameters for latent variable

%          vari       inatt          B    lambda    vara    vars     phi    tau_phi 
lb =      [1e-6       0              2    -10       1e-6    1e-6     0.1    0.01]';
ub =      [10.0       1.0            100   10       1000    100      1.2    1.0]';
fit_vec = [false(1,1);false(1,1);    true(4,1);                      true(2,1)];

%these will be the defaults it fit_vec is false for an entry
if use.spikes
    x0 =  [1e-6       0.0,           20,   0.0,     50,     1.0,      0.8,  0.02]';   
else
    x0 =  [1e-5       0.5,           10,   2.0,     50,     0.5,      0.8,  0.02]';
end

%% Data likelihood parameters

%if using choice data in the optimization
if use.choice 
    %concatenate all values onto the end of existing vectors
    lbd = -1; ubd = 1;
    x0 = cat(1,x0,0);
    lb = cat(1,lb,lbd); ub = cat(1,ub,ubd);
    fit_vec = cat(1,fit_vec,true);
    
end

%....if using neural data
if use.spikes    
    
    %choose bounds for each parameter type, depending on the firing rate
    %function selected
    switch dimy
        case 4
            %      a        b       c       d
            lby1 = [0      0       -5.0    -5.0]';
            uby1 = [100    100      5.0     5.0]';
            %lby1 = [0      0       -inf    -inf]';
            %uby1 = [inf    inf      inf     inf]';
        case 3
            lby1 = [0      -10.0    -5.0]';
            uby1 = [100     10.0     5.0]';
        case 2
            lby1 = [-10.0    -5.0]';
            uby1 = [10.0     5.0]';
    end
    
    %make an initial guess of the neural tuning curve parameters using the
    %stimulus data
    y0 = init_x0y(data,lby1,uby1);
    %reshape into a vector
    lby = reshape(repmat(lby1,1,N)',[],1); uby = reshape(repmat(uby1,1,N)',[],1);
    
    %concatenate all values onto the end of existing vectors
    x0 = cat(1,x0,reshape(y0',[],1));
    lb = cat(1,lb,lby); ub = cat(1,ub,uby);    
    fit_vec = cat(1,fit_vec,true(dimy*N,1));
    
end

%%

%x = lb + (ub-lb)*0.5*(1+tanh(x));
x0 = atanh(((x0 - lb)./((ub-lb).*0.5)-1));

%%

%filter out bounds that are not being fit
%lb = lb(fit_vec); ub = ub(fit_vec);

%use generative parameters to initialize x
if ~isempty(xgen); x0 = xgen; end
