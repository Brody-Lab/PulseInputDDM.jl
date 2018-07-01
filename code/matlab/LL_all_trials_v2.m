function varargout = LL_all_trials_v2(x,data,dt,n,dimz,dimd,dimy,model_type,...
    N,fr_func,posterior)

%MAIN FUNCTION ROUTINE FOR COMPUTING THE LOGLIKELIHOOD

%inputs: 1. x: vector of all parameters
%        2. data: struct containing all data for performing parameter
%           learning
%        3. dt: temporal bin width
%        4. n: number of spatial bins for discrete approximation of latent
%           distribution
%        5. dimz: number of parameters in the latent variable
%        6. dimd: number of choice data parameters (normally 1)
%        7. dimy: number of spiking data parameters (number of parameters
%           in tuning curve)
%        8: model_type: string which determines the types of data to use
%        9: N: number of neurons, if spiking data is used
%        10: fr_func: function handle for the tuning curve
%        11: posterior: flag to compute or not compute the posterior over
%           the latent

%outputs:
%if computing posterior : posterior, bin center locations, forward-pass density
%else: neg LL

%% define some default parameters

%completes parameter vector with static parameters or simply reassigns
%parameters vector as parameters that are being optimized

%compute the posterior over the latent variable, default is false, since
%computing the posterior is not necessary for doing gradient descent and it
%is costly to compute
if nargin < 11 || isempty(posterior)
    posterior = false;
end

%% construct various things for efficiently computing the log-likelihood

tri = numel(data); %number of trials

%convert the entries of x into variables that are easier to read
vari = x(1); inatt = x(2); B = x(3); lambda = x(4); vara = x(5);

dx = 2*B/(n-2); %spatial bin width of latent variable
xc = [linspace(-(B+dx/2),-dx,(n-1)/2),0,linspace(dx,(B+dx/2),(n-1)/2)]; %location of bin centers

%compute the latent variable transition matrix using
%the "settling" method of Brunton et. al.
M = M_settling_v2(xc, vara * dt, lambda, 0, dt, dx);

%define the initial distribution of the latent variable
P = zeros(n,1); P(xc == 0) = (1 - inatt); P([1,n]) = P([1,n]) + inatt/2;

%set the initial variance
M0 = M_settling_v2(xc, vari, 0, 0, 0, dx); P = M0 * P;

%% construct certain things relevant to the observation likelihood term

%this will be different depending on the type of observation data

%using choice data
if any(strcmp(model_type,{'choice','joint'}))
    bias = x(9); %bias term
    nbinsL = ceil((B+bias)/dx); %number of COMPLETE bins to the LEFT of the bias
    Sfrac = (1/dx) * (bias - (-(B+dx)+nbinsL*dx)); %fraction of bin that is bisected by that bias that is left of the bias
    
else
    %not using choice data, so these are not needed
    nbinsL = []; Sfrac = [];
end

%using neural data
if any(strcmp(model_type,{'spikes','joint'}))
    %ALSO using choice data
    if any(strcmp(model_type,{'joint'}))
        switch dimy
            %sigmoid
            case 4
                a = x([1:N]+dimd+dimz); b = x([1:N]+dimd+dimz+N); c = x([1:N]+dimd+dimz+2*N); d = x([1:N]+dimd+dimz+3*N);
                %compute spike count expectation at every bin center
                lambda = fr_func([a,b,c,d],xc);
        end
    else
        %map elements of x to tuning curve parameters for all N neurons
        %this will depend on the number of parameters for the tuning curves
        switch dimy
            %sigmoid
            case 4
                a = x([1:N]+dimz); b = x([1:N]+dimz+N); c = x([1:N]+dimz+2*N); d = x([1:N]+dimz+3*N);
                %compute spike count expectation at every bin center
                lambda = fr_func([a,b,c,d],xc);
                %softplus
            case 3
                a = x([1:N]+dimz); c = x([1:N]+dimz+N); d = x([1:N]+dimz+2*N);
                %compute spike count expectation at every bin center
                lambda = fr_func([a,c,d],xc);
                %exponential
            case 2
                c = x([1:N]+dimz); d = x([1:N]+dimz+N);
                %compute spike count expectation at every bin center
                lambda = fr_func([c,d],xc);
        end
    end
    
else
    %not using neural data, so these are not necessary
    lambda = [];
end

%% compute the log-likelihood for each trial separately, within a parfor for efficiency

% if computing posterior or not, different outputs will need to be called
if posterior
    
    %empty cell for the posterior for each trial
    post = cell(tri,1);
    %empty cell for the discrete-approximation of the latent variable on
    %the forward-pass (which can be useful elsewhere)
    alpha = cell(tri,1);
    
    %loop over trials
    parfor i = 1:tri
        [post{i},alpha{i}] = LL_single_trial(data(i),x(4:8),P,M,dx,xc,...
            Sfrac,nbinsL,lambda(:,data(i).N),n,dt,model_type,posterior);
    end
    
    %define outputs
    %outputs: posterior, bin center locations, forward-pass density
    varargout{1} = post;
    varargout{2} = xc;
    varargout{3} = alpha;
    
else
    
    %initialize log-likelihood
    LL = 0;
    
    %loop over trials
    parfor i = 1:tri
        LL = LL + LL_single_trial(data(i),x(4:8),P,M,dx,xc,...
            Sfrac,nbinsL,lambda(:,data(i).N),n,dt,model_type,posterior);
    end
    
    LL = -real(LL); %negative log liklihood of all trials
    %outputs: sum of negative log likelihood
    varargout{1} = LL;
    
end

end

%% subfunctions

function varargout = LL_single_trial(data,x,P,M,dx,xc,Sfrac,nbinsL,...
    lambda,n,dt,model_type,posterior)

%main subfunction routine for computing the log-likelihood and/or posterior of
%an indivdual trial

%inputs: 1. data: struct containing all data for this trial for performing parameter
%           learning
%        2. x: any remaining parameters necessary for constructing various
%           objects for this trial
%        3. P: discrete latent variable prior, i.e. P(a(0))
%        4. M: latent variable transition matrix, i.e. P(a_{t+1} | a_t)
%        5. dx: spatial bin width
%        6. xc: bin center locations
%        7. Sfrac: fraction of bias-bisected bin (choice data only)
%        8: nbinsL: number of COMPLETE bins to the left of bias (choice
%           data only)
%        9: lambda: spike count expectation (neural data only)
%        10. n: number of spatial bins for discrete approximation of latent
%           distribution
%        11. dt: temporal bin width
%        12: model_type: string which determines the types of data to use
%        13: posterior: flag to compute or not compute the posterior over
%           the latent

%outputs:
%if computing posterior : posterior, forward-pass density
%else: LL

%%

%default parameters
if nargin < 13 || isempty(posterior)
    posterior = false;
end

%convert the entries of x into variables that are easier to read
lambda_drift = x(1); vara = x(2); vars = x(3); phi = x(4); tau_phi = x(5);
%number of temporal bins on this trial
nT = data.nT;

%compute adapted magnitude of clicks
[La, Ra] = adapted_clicks(data.leftbups, data.rightbups, phi, tau_phi);

%if using choice data, construct conditional likelihood of choice data
if any(strcmp(model_type,{'choice','joint'}))
    %direction of animal choice on this trial
    pokedR = data.pokedR;
    %compute liklihood of the choice, given a_T: P(d | a_T)
    Pd = [~pokedR * ones(nbinsL,1); ...
        ~pokedR * Sfrac + pokedR * (1 - Sfrac); ....
        pokedR * ones(n-(nbinsL+1),1)];
    
end

%if using neural data, construct conditional likelihood of neural data for all
%times bins
if any(strcmp(model_type,{'spikes','joint'}))
    %we assume conditional independence amongst the neurons so Py(t) = Pi_i^N
    %Py_i(t) = Pi_i^N poiss(lambda_i(t) * dt);
    Py = exp(data.spike_counts * log(lambda'*dt) ...
        - ones(nT,size(lambda,2))*lambda'*dt ...
        - gammaln(data.spike_counts+1)*ones(size(lambda,2),n))';
    
end

%empty vector for saving p(X_t), for normalizing distribution as
%computation proceeds
c = NaN(nT,1);
if posterior; alpha = zeros(n,nT); end %for saving forward pass terms

%FORWARD PASS, loop over time
for t = 1:nT
    
    %if any left pulses in this time bin...
    if any(t == data.hereL)
        %...sum all left pulses in that time bin.
        sL = sum(La(t == data.hereL));
        
    else
        sL = 0;
        
    end
    
    %if any right pulses in this time bin...
    if any(t == data.hereR)
        %...sum all right pulses in that time bin.
        sR = sum(Ra(t == data.hereR));
        
    else
        sR = 0;
        
    end
    
    %compute variance and mean of Gaussian due to input
    %variance will be the sum of the left and right adapted magnitude
    %mean will be the difference of the left and right adapted magnitude
    var = vars * (sL + sR); mu = -sL + sR;
    
    %if var > 0, i.e. was there a either L or R pulse?
    if var > 0
        M_sett = M_settling_v2(xc, var + vara*dt, lambda_drift, mu/dt, dt, dx);
        %pulse diffuse and shift
        P = M_sett * P;
    else      
        %evolve the latent variable distribution
        P = M * P;
    end
    
    %if using spike data
    if any(strcmp(model_type,{'spikes','joint'}))
        %element-wise multiplication by the spike observation likelihood
        P = P .* Py(:,t); % P(a_t | a_{t-1}) * P(y_t | a_t)
        
    end
    
    %if using choice data and at the end of the trial
    if any(strcmp(model_type,{'choice','joint'})) && t == nT
        %element-wise multiplication by the choice observation likelihood
        P = P .* Pd; %P(a_t | a_{t-1}) * P(d | a_t)
        
    end
    
    %P(x_t)
    %added eps incase sum(P) = 0 so that we don't divide P by 0
    c(t) = sum(P) + eps;
    %scale for numerical stability. see section 13.2.4 "Scaling factors" of
    %Bishop for an explanation of why this is necessary in these sort of
    %sequential models (http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
    P = P/c(t);
    
    %save P(a_t) to compute full posterior over the latent variable
    if posterior; alpha(:,t) = P; end
    
end

%log likelihood
%log(Pi_t^nT c_t) = log(P(X)) = sum_t^nT c_t
LL = sum(log(c));

%if computing the posterior over the latent variable, we need to run the
%backward pass
if posterior
    
    beta = zeros(n,nT); %for saving backward pass terms
    P = ones(n,1); %initialze backward pass with all 1's
    
    beta(:,nT) = P;
    
    %BACKWARD PASS, loop backwards in time
    for t = nT-1:-1:1
        
        %if using spike data
        if any(strcmp(model_type,{'spikes','joint'}))
            %update posterior based on spike likelihood term at time t
            P = P .* Py(:,t+1); % P(a_t | a_{t-1}) * P(y_t | a_t)
            
        end
        
        if any(strcmp(model_type,{'choice','joint'})) && t+1 == nT
            %Compute likelihood given data
            P = P .* Pd; %P(a_t | a_{t-1}) * P(d | a_t)
            
        end
        
        %drift and diffuse only, notice transpose, due to backward movement
        P = M' * P;
        
        %this could be made more efficient by saving left and right pulse
        %information from forward pass
        %left pulses
        if any(t+1 == data.hereL)
            sL = sum(La(t+1 == data.hereL));
            
        else
            sL = 0;
            
        end
        
        %right pulses
        if any(t+1 == data.hereR)
            sR = sum(Ra(t+1 == data.hereR));
            
        else
            sR = 0;
            
        end
        
        %compute variance and mean for input Gaussian
        var = vars * (sL + sR); mu = -sL + sR;
        
        %if either a left or right pulse occured
        if var > 0
            M_sett = M_settling_v2(xc, var, 0, mu/dt, dt, dx);
            %pulse diffuse and shift
            P = M_sett' * P;
            
        end
        
        P = P/c(t+1); %normalize based on normalizing factor computed on forward pass
        beta(:,t) = P; %save term from backward pass
        
    end
    
    %posterior at each time point, given all data is product of forward and
    %backward pass
    varargout{1} = alpha .* beta;
    varargout{2} = alpha; %forward pass
    
else
    
    varargout{1} = LL; %log likelihood of this trial
    
end

end