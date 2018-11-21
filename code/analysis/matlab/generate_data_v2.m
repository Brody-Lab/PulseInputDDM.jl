function [data,xgen] = generate_data_v2(ratname,model_type,sessid,in_pth,xgen,more_tri_flag,num_tri,dt)

global dimz dimy N lby1 uby1

if nargin < 6 || isempty(more_tri_flag)
    more_tri_flag = false;
    num_tri = [];
end

%% Load real data

%empty array
data = struct('pokedR',[],'spike_counts',[],'nT',[],'hereL',[],'hereR',[],'hereLR',[],'N',[],...
    'T',[],'leftbups',[],'rightbups',[]);

for i = 1:numel(sessid)
    load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{i})),'rawdata');
    %process rawdata
    data = package_data_v2(rawdata,data,[],dt,model_type);
end

%for plotting psth of data
%[~,psth,conds] = psth_cta(data);
%plot_psth(conds,psth,data);

%% Use real data to generate parameters for neurons
%generate data
if nargin < 5 || isempty(xgen)
    
            %vari,     lapse,   B,    lambda,  vara,   vars,    phi,   tau_phi
    %xgen =  [1e-6,     0.0,     40,   6,       125,    0.07,    1.04,   0.13]'; %close to T036 results from spikes
    xgen =  [1e-6,     0.11,     5.3,   0.002,   1,    1.7,    0.37,   0.04]';  %close to T036 results from behavior
    %xgen = [1e-6,       0.0       10.0    -0.5    20    1.0     0.8     0.05]';

    if use.choice
        
        xd = -0.66;
        xgen = cat(1,xgen,xd);
        
    end
    
    if use.spikes
        
        %estimate the tuning based on the data and use that as the
        %generative parameter
        xy = init_x0y(data,lby1,uby1);
        xgen = cat(1,xgen,reshape(xy',[],1));
        
    end
    
end

%% Generate fake data

%5/5 commented out to generate many random samples of data
%rng(1);

%empty array
data = struct('pokedR',[],'spike_counts',[],'nT',[],'hereL',[],'hereR',[],'hereLR',[],'N',[],...
    'T',[],'leftbups',[],'rightbups',[]);

ns_per_sess = NaN(numel(sessid,1));

for i = 1:numel(sessid)
    
    load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{i})),'rawdata');
    
    %might need to fix this in case of using joint data
    if strcmp(model_type,'spikes')
        %for making sure correct parameters are use for neurons
        if isempty(data(1).T)
            N0 = 0;
        else
            %for adding more trials from a different session
            N0 = data(end).N(end);
        end
        ns_per_sess(i) = size(rawdata(1).St,2);
        N2 = N0 + size(rawdata(1).St,2);
        
        %changed 2/28, to accomodate varioud fr_func, but didn't 100% check
        %OK.
        switch dimy
            case 4
                ahat = xgen([N0+1:N2]+dimz); bhat = xgen([N0+1:N2]+dimz+N);
                chat = xgen([N0+1:N2]+dimz+2*N); dhat = xgen([N0+1:N2]+dimz+3*N);
                xgen2 = cat(1,xgen(1:dimz),ahat,bhat,chat,dhat);
            case 3
                %a = x([1:N]+dimz); c = x([1:N]+dimz+N); d = x([1:N]+dimz+2*N);
            case 2
                chat = xgen([N0+1:N2]+dimz); dhat = xgen([N0+1:N2]+dimz+N);
                xgen2 = cat(1,xgen(1:dimz),chat,dhat);
        end
                
    elseif strcmp(model_type,'choice')
        
        xgen2 = xgen;
        N0 = 0;
        
    end
    
    %generate new dataset
    [rawdata] = sample_model(rawdata,xgen2,model_type);
    %process rawdata
    data = package_data_v2(rawdata,data,[],dt,model_type);
    
end

%for adding additional trials
if more_tri_flag
    
    while numel(data) < num_tri
        
        i = randsample(numel(sessid),1);
        
        load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{i})),'rawdata');
            
        %might need to fix this in case of using joint data
        if strcmp(model_type,'spikes')   
            %added this to find the right neural index for the new dataset,
            %since these neurons are already included in there.
            N0 = sum(ns_per_sess(1:i-1));
            N2 = N0 + size(rawdata(1).St,2);
            ahat = xgen([N0+1:N2]+dimz); bhat = xgen([N0+1:N2]+dimz+N);
            chat = xgen([N0+1:N2]+dimz+2*N); dhat = xgen([N0+1:N2]+dimz+3*N);
            xgen2 = cat(1,xgen(1:dimz),ahat,bhat,chat,dhat);
            
        elseif strcmp(model_type,'choice')
            
            xgen2 = xgen;
            
        end
        
        %generate new dataset
        [rawdata] = sample_model(rawdata,xgen2,model_type);
        %process rawdata
        data = package_data_v2(rawdata,data,N0,dt,model_type);
        
    end
    
end

%for total number of neurons, added so when save data, there is correct
%field there for julia
for i = 1:numel(data)
    data(i).Ntotal = N;
end

%save the new dataset
%if use.spikes
%    save(fullfile(save_pth,sprintf('data_w_spikes_%s.mat',ratname)),'rawdata','xgen');
%elseif use.choice
%    save(fullfile(save_pth,sprintf('chrono_%s_rawdata.mat',ratname)),'rawdata','xgen');
%end

end


