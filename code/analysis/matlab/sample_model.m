function [rawdata,A,lambda,spikes] = sample_model(rawdata,x,model_type)

global dimz dimy fr_func

tri = numel(rawdata);
if any(strcmp(model_type,{'spikes','joint'}))
    N = size(rawdata(1).St,2);
end
dt = 1e-4;
%save old rawdata, in case we want to look at which neurons spikes on which
%trials
old = rawdata;
rawdata = struct('T',[],'leftbups',[],'rightbups',[],'St',[],'pokedR',[]);

%just use the originial rawdata
for j = 1:tri
    rawdata(j).T = old(j).T;
    rawdata(j).leftbups = old(j).leftbups;
    rawdata(j).rightbups = old(j).rightbups;
end

tmax = max(cell2mat({rawdata.T}));

%latent variable model
vari = x(1); %initial variance
inatt = x(2); %lapse rate (choose a random choice)
B = x(3); %bound height
lambda = x(4); %1/tau, where tau is the decay timescale
vara = x(5); %diffusion variance
vars = x(6); %stimulus variance
phi = x(7); %click adaptation magnitude
tau_phi = x(8); %adaptation decay timescale

if strcmp(model_type,'choice')
    bias = x(9);
end

T = zeros(tri,1);
diffLR = NaN(tri,round(tmax/dt));
sumLR = NaN(tri,round(tmax/dt));

for i = 1:tri
    
    T(i) = ceil(rawdata(i).T/dt) * dt;  % number of timesteps
    tvec = 0:dt:T(i)-dt;
    
    [clicks_L, clicks_R] = adapted_clicks(rawdata(i).leftbups, rawdata(i).rightbups, phi, tau_phi);
    [difflr, sumlr] = click_inputs(tvec, rawdata(i).leftbups, rawdata(i).rightbups, clicks_L, clicks_R);
    
    diffLR(i,1:numel(tvec)) = difflr;
    sumLR(i,1:numel(tvec)) = sumlr;
    
end

%rng(1);

r1 = randn(tri,1);
r2 = rand(tri,1);
r3 = randn(tri,1);

%initialize with appropriate variance
a = sqrt(vari) * r1; %initialize particles
vec = r2 < inatt; %pick certain number of particles that are initalized across the boundary randomly
a(vec) = B * sign(r3(vec)); %evenly distribute them at -B and B

A = NaN(tri,max(round(T/dt)));

for t = 1:max(round(T/dt))
    
    within_time = t <= round(T/dt);
    %if particles crosses boundary or time has elapsed, it is no longer
    %integrated
    go = a < B & a > -B & within_time;
 
    %one time step, 4 terms: 1- decay,
    %2- input,
    %3- diffusion noise,
    %4-stimulus noise
    a(go,1) = a(go,1) + (dt*lambda) * (a(go,1))...
        + diffLR(go,t)...
        + sqrt(vara * dt) * randn(sum(go),1) ...
        + sqrt(sumLR(go,t) * vars) .* randn(sum(go),1);
    
    %push particles that went over the border to the border.
    a(a > B) = B; a(a < -B) = -B;
    
    A(within_time,t) = a(within_time);
    
end

if strcmp(model_type,'spikes')
    switch dimy
        case 4
            ahat = x([1:N]+dimz); bhat = x([1:N]+dimz+N); chat = x([1:N]+dimz+2*N); dhat = x([1:N]+dimz+3*N);
            xy = [ahat,bhat,chat,dhat];
        case 3
            ahat = x([1:N]+dimz); chat = x([1:N]+dimz+N); dhat = x([1:N]+dimz+2*N);
            xy = [ahat,chat,dhat];
        case 2
            chat = x([1:N]+dimz); dhat = x([1:N]+dimz+N);
            xy = [chat,dhat];
    end
    
    %changed 2/28, to accomodate varioud fr_func, but didn't 100% check
        %OK.
    lambda = permute(reshape(fr_func(xy,reshape(A,1,[])),tri,[],N),[3,2,1]);
    spikes = poissrnd(lambda*dt);
    
end

for i = 1:tri
    
    if any(strcmp(model_type,{'joint','choice'}))
        rawdata(i).pokedR = a(i) > bias;        
    else
        rawdata(i).pokedR = a(i) > 0;        
    end
    
    if any(strcmp(model_type,{'joint','spikes'}))
      
        %temp = poissrnd(fr_func(xy,A(i,:))'*dt);
        %temp = squeeze(spikes(:,:,i));
        
        for j = 1:N
            %initialze as no spikes
            rawdata(i).St{j} = zeros(0,1);
            %find any spikes
            tidx = find(spikes(j,:,i) ~= 0 & ~isnan(spikes(j,:,i)));
            %loop over, adding the number of spikes that happened
            for k = 1:numel(tidx)
                rawdata(i).St{j} = cat(1,rawdata(i).St{j}, ...
                    repmat(dt * (tidx(k) - 1),spikes(j,tidx(k),i),1)); %repeat number of times as there is spikes in that bin, subtract by 1 to make base zero and multiply by dt to set spike time in ms
            end
        end
        
    else
        rawdata(i).St = [];
    end
    
end

end