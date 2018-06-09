function [data,N] = load_data_v2(ratname,sessid,in_pth,dt,model_type)

%empty struct for "packaged" data
data = struct('pokedR',[],'spike_counts',[],'nT',[],'hereL',[],'hereR',[],...
    'hereLR',[],'N',[],'T',[],'leftbups',[],'rightbups',[]);

%loop over sessions
for i = 1:numel(sessid)
    load(fullfile(in_pth,sprintf('%s_%s.mat',ratname,sessid{i})),'rawdata');
    %process rawdata
    %changed 2/14 to include dt as input, so can refilter data with smaller
    %dt for computing firing rates
    [data,N] = package_data_v2(rawdata,data,[],dt,model_type);
end

%for total number of neurons, added so when save data, there is correct
%field there for julia
for i = 1:numel(data)
    data(i).Ntotal = N;
end
