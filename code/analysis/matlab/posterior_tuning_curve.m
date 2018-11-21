function [mSC,xe2] = posterior_tuning_curve(xf,fit_vec,data,xc,posterior,dt)

global N dimz

tri = numel(data); %number of trials

x = NaN(1,dimz); %unpack parameters

dimz2 = sum(fit_vec(1:dimz)); %how many parameters were for the latent
x(1,logical(fit_vec(1:dimz))) = xf(1:dimz2);

nbins2 = 20;
mSC = zeros(nbins2,N);

for j = 1:N
                
    spikes = []; mpost = []; 
    %diffLRcat = [];
    %diffLR = cell(tri,1);
    
    for i = 1:tri
        
        if any(data(i).N == j)
            
            spikes = cat(1,spikes,data(i).spike_counts(:,data(i).N == j));
            mpost = cat(1,mpost,posterior{i}' * xc');
            %t = 0:dt:data(i).nT*dt;
            %diffLR{i} = cumsum(-histcounts(data(i).leftbups,t) + ...
            %    histcounts(data(i).rightbups,t));
            %diffLRcat = cat(1,diffLRcat,reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
            %    histcounts(data(i).rightbups,t)),[],1));
            
        end
        
    end
    
    xe2 = linspace(-x(3),x(3),nbins2 + 1);
    
    for i = 1:numel(xe2)-1
        mSC(i,j) = mean(spikes(mpost >= xe2(i) & mpost < xe2(i+1)))/dt;
    end
    
end
