function x0y = init_x0y(data,lb,ub)

%1/18: this function should correctly deal with data containing more than
%one session's worth of neurons

global dt N dimy fr_func

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
    b = temp{i}(:,1)'/[temp{i}(:,2),ones(numel(temp{i}(:,2)),1)]';
    
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
    %estimate tuning curve
    x0y(:,i) = lsqcurvefit(fr_func,x0,temp{i}(:,2),temp{i}(:,1)',lb,ub,options);     
    
    %[bhat,fitinfo] = lassoglm(temp{1}(:,2),temp{1}(:,1)*dt,'poisson','Alpha',alpha,...
    %    'CV',10,'Options',statset('UseParallel',true));
    
    %usebhat = bhat(:,fitinfo.IndexMinDeviance);    
    %frhat = exp(usebhat' * X' + fitinfo.Intercept(fitinfo.IndexMinDeviance));
    
    if 0
        nbins2 = 20;
        figure;clf;set(gcf,'color','w','Menubar','none');
        
        x_lim1 = min(temp{i}(:,2));
        x_lim2 = max(temp{i}(:,2));
        xe2 = linspace(x_lim1,x_lim2,nbins2 + 1);
        xc2 = xe2(1:nbins2)+mean(diff(xe2))/2;
        
        tcurve = fr_func(x0y(:,i)',xc2);
        
        mSC = zeros(nbins2,1);
        for ii = 1:numel(xe2)-1
            mSC(ii) = mean(temp{i}(temp{i}(:,2) >= xe2(ii) & temp{i}(:,2) < xe2(ii+1)),1);
        end
        
        plot(xc2,tcurve,'color','r','LineWidth',2,'Color','r');hold on;
        scatter(xc2,mSC,'x','MarkerFaceColor','k','MarkerEdgeColor','k');
        
        set(gca,'box','off','ylim',[0 max(tcurve)]);
        xlabel('post. mean');ylabel('fr (hz)');
        drawnow;
    end
      
end