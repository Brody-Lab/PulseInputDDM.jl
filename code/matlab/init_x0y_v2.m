function x0y = init_x0y_v2(data)

%1/18: this function should correctly deal with data containing more than
%one session's worth of neurons

global dt N dimy fr_func

%%

%empty cell for spike counts and diff clicks for each neuron
temp = cell(N,1);
tri = numel(data);

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

%loop over neurons
for i = 1:N
    
    %linear regression to find slope of tuning
    bs(:,i) = temp{i}(:,1)'/[temp{i}(:,2),ones(numel(temp{i}(:,2)),1)]';
    b = bs(:,i);
    
    %use 3- or 4- parameter rate funcdtion
    switch dimy
        case 4
            %sigmoid
            x0 = [min(temp{i}(:,1)),max(temp{i}(:,1)),b(1),0];
        case 3
            %softplus
            x0 = [min(temp{i}(:,1)),b(1),0];
        case 2
            %exp
            x0 = [b(1),0];
    end
    lb = [0      0       -inf    -inf]';
    ub = [inf    inf      inf     inf]';
    %estimate tuning curve
    x0y(:,i) = lsqcurvefit(fr_func,x0,temp{i}(:,2),temp{i}(:,1)',lb,ub,options);     
      
end

%%

nbins2 = 20;
xc = linspace(min(temp{2}(:,2)),max(temp{2}(:,2)),n);
mSC = zeros(nbins2,N);
temp2 = NaN(N,n);

for j = 1:N
                
    spikes = []; 
    
    for i = 1:tri
        
        if any(data(i).N == j)
            
            spikes = cat(1,spikes,data(i).spike_counts(:,data(i).N == j));
            
        end
        
    end
    
    xe2 = linspace(min(temp{j}(:,2)),max(temp{j}(:,2)),nbins2 + 1);

    for i = 1:numel(xe2)-1
        mSC(i,j) = mean(spikes(temp{j}(:,2) >= xe2(i) & temp{j}(:,2) < xe2(i+1)))/dt;
    end
    
    temp2(j,:) = fr_func([x0y(1:2,j)',abs(x0y(3,j)),x0y(4,j)],xc);
    temp3(j,:) = fr_func([xy(j,1:2),abs(xy(j,3)),xy(j,4)],xc);
    
end

%%

figure;hold on;
for i = 1:N
    subplot(3,4,i);hold on;
    scatter(xe2(1:end-1)+mean(diff(xe2))/2,mSC(:,i),'x');
    plot(xc,temp2(i,:));
    plot(xc,temp3(i,:));
end
