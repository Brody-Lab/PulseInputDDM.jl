clearvars;

load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/julia/spikes/157201_157357_157507_168499/17828422/data_T036_157201_157357_157507_168499_spikes.mat');
%load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/julia/spikes/157201_157357_157507_168499/17813002/julia_T036_157201_157357_157507_168499_spikes.mat');

%xy = reshape(xf(9:end),[],4);

n = 203; dt = 2e-2; dimy = 4; N = 12;

%fr_func = @(x,z)bsxfun(@plus,x(:,1),bsxfun(@rdivide,x(:,2),...
%    (1 + exp(bsxfun(@plus,-x(:,3) * z,x(:,4))))))' + eps;

fr_func = @(x,z) x(1).^2 + x(2).^2./(1 + exp(-x(3) * z + x(4)));

%%

nbins2 = 20;
%empty cell for spike counts and diff clicks for each neuron
temp = cell(N,1);
tri = numel(data);
temp2 = NaN(N,nbins2);

%loop over trials
for i = 1:tri
    
    %compute the cumulative diff of clicks
    t = 0:dt:data(i).nT*dt;
    diffLR = reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
        histcounts(data(i).rightbups,t)),[],1);
            
        for j = 1:numel(data(i).N)
            temp{data(i).N(j)} = cat(1,temp{data(i).N(j)},cat(2,data(i).spike_counts(:,j)/dt,diffLR));
        end
            
end

x0y = NaN(dimy,N);
options = optimoptions(@lsqcurvefit,'Display','off');

%xc = linspace(min(temp{2}(:,2)),max(temp{2}(:,2)),n);
mSC = zeros(nbins2,N);
xe2 = linspace(min(temp{1}(:,2)),max(temp{1}(:,2)),nbins2 + 1);
xc = xe2(1:end-1)+mean(diff(xe2))/2;

%loop over neurons
for i = 1:N
    
    %linear regression to find slope of tuning
    %bs(:,i) = temp{i}(:,1)'/[temp{i}(:,2),ones(numel(temp{i}(:,2)),1)]';
    %b = bs(:,i);
    
    %x0 = [min(temp{i}(:,1)),max(temp{i}(:,1)),b(1),0];
    x0 = [0,10,0,0];
    %lb = [0      0       -inf    -inf]';
    %ub = [inf    inf      inf     inf]';
    lb = []; ub = [];
    %estimate tuning curve
    x0y(:,i) = lsqcurvefit(fr_func,x0,temp{i}(:,2),temp{i}(:,1),lb,ub,options); 
    
    temp2(i,:) = fr_func([x0y(1:2,i)',(x0y(3,i)),x0y(4,i)],xc');
      
end

%%

for j = 1:N
                  
    for i = 1:numel(xe2)-1
        mSC(i,j) = mean(temp{j}(temp{j}(:,2) >= xe2(i) & temp{j}(:,2) < xe2(i+1)),1);
    end
    
    %temp2(j,:) = fr_func([x0y(1:2,j)',(x0y(3,j)),x0y(4,j)],xc');
    %temp3(j,:) = fr_func([xy(j,1:2),(xy(j,3)),xy(j,4)],xc);
    
end

%% 

x0y(1:2,:) = x0y(1:2,:).^2;

%%

x0y_old = load('/Users/briandepasquale/Dropbox/spike-data_latent-accum/data/results/new2/157201_157357_157507_168499/T036_157201_157357_157507_168499_spikes.mat','x0');
x0y_old = reshape(x0y_old.x0(9:end),N,4)';

%%

figure;hold on;
for i = 1:N
    subplot(3,4,i);hold on;
    scatter(xc,mSC(:,i),'x');
    plot(xc,temp2(i,:));
    %plot(xc,temp3(i,:));
end
