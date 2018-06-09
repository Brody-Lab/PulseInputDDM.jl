function [data,N] = package_data_v2(rawdata,data,N0,dt,model_type)

%changed 2/14 to include dt as input, so can refilter data with smaller
%dt for computing firing rates
%added N0 input in case adding new "fake data"

if nargin < 3  || isempty(data)
    
    %empty array
    data = struct('pokedR',[],'spike_counts',[],'nT',[],'hereL',[],'hereR',[],'hereLR',[],'N',[],...
        'T',[],'leftbups',[],'rightbups',[]);
    
end

if isempty(data(1).T)    
    tri0 = 0;
    
else
    %for adding more trials from a different session
    tri0 = numel(data);
    
end

if nargin < 4 || isempty(N0) && any(strcmp(model_type,{'spikes','joint'}))
    if isempty(data(1).T)        
        N0 = 0;
        
    else        
        %for adding more trials from a different session
        N0 = data(end).N(end);
        
    end
end

for i = 1:numel(rawdata)
    
    data(i+tri0).T = rawdata(i).T;
    data(i+tri0).leftbups = rawdata(i).leftbups;
    data(i+tri0).rightbups = rawdata(i).rightbups;
    data(i+tri0).nT = ceil(rawdata(i).T/dt);
    t = 0:dt:data(i+tri0).nT*dt;
    data(i+tri0).hereL = qfind(t, rawdata(i).leftbups);
    data(i+tri0).hereR = qfind(t, rawdata(i).rightbups);
    data(i+tri0).pokedR = rawdata(i).pokedR;
    
    if any(strcmp(model_type,{'spikes','joint'}))   
        tempN = []; temp = [];        
        %loop over neurons
        for j = 1:size(rawdata(1).St,2)            
            %which neurons spiked
            tempN = cat(1,tempN,j+N0);
            %bin the spikes
            temp = cat(2,temp,reshape(histcounts(rawdata(i).St{j},t),[],1));            
        end
        
        data(i+tri0).spike_counts = temp; data(i+tri0).N = tempN;
        
    end
    
end

if any(strcmp(model_type,{'spikes','joint'}))
    N = N0 + size(rawdata(1).St,2);
else
    N = 0;
end

