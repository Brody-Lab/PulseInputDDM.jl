function [data,use,N] = load_data(ratname,model_type,sessid,in_pth,dt)

%create empty struct indicating what type of data to use
use = struct('choice',false,'spikes',false);

%which type of data to use to fit model
switch model_type
    case 'choice'       
        use.choice = true;
        
    case 'spikes'        
        use.spikes = true;
        
    case 'joint'       
        use.choice = true;
        use.spikes = true;
        
end

%empty struct for "packaged" data
data = struct('pokedR',[],'spike_counts',[],'nT',[],'hereL',[],'hereR',[],...
    'hereLR',[],'N',[],'T',[],'leftbups',[],'rightbups',[]);

%loop over sessions
for i = 1:numel(sessid)
    if strcmp(sessid{1},'1') && use.choice == true;
        load(fullfile(in_pth,sprintf('chrono_%s_rawdata.mat',ratname)),'rawdata');
    else
        load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{i})),'rawdata');
    end
    %process rawdata
    %changed 2/14 to include dt as input, so can refilter data with smaller
    %dt for computing firing rates
    [data,N] = package_data(rawdata,use,data,[],dt);
end

%for total number of neurons, added so when save data, there is correct
%field there for julia
for i = 1:numel(data)
    data(i).Ntotal = N;
end
