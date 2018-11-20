clearvars;

path = '/Users/briandepasquale/Projects/inProgress/spike-data_latent-accum/data/koay/';

load(fullfile(path,'k56_20160826_PM_500um_139mW_zoom2p2.modeling.mat'));
dt = 1/15;

%%

%empty array
rawdata = struct('pokedR',[],'St',[],'cell',[],...
    'T',[],'leftbups',[],'rightbups',[],'correct_dir',[], 'sessid',[],...
    'nT', [], 'std', [], 'binned_leftbups', [], 'binned_rightbups', [], ...
    'position', []);
    

for i = 1:length(score.trial);
    
    %which way the animal choose
    if score.trial(i).choice == 1
        rawdata(i).pokedR = false;
    elseif score.trial(end).choice == 2
        rawdata(i).pokedR = true;
    end
    
    %which way was supposed to choose
    if score.trial(i).trialType == 1
        rawdata(i).correct_dir = false;
    elseif score.trial(i).trialType == 2
        rawdata(i).correct_dir = true;
    end
    
    %event times
    temp = score.dataDFF(score.trial(i).fCueEntry:score.trial(i).fMemEntry,:);
    count = 1;
    for j = 1:size(temp,2)
        if ~isnan(temp(1,j))
            rawdata(i).St{count} = score.dataDFF(score.trial(i).fCueEntry:score.trial(i).fMemEntry,j);
            rawdata(i).cell{count} = j;
            count = count + 1;
        end
    end
    %trial length
    rawdata(i).position = event.position(score.trial(i).fCueEntry:score.trial(i).fMemEntry,2);
    rawdata(i).nT = (score.trial(i).fMemEntry - score.trial(i).fCueEntry + 1);
    rawdata(i).T = dt * rawdata(i).nT;
    %cell id
    rawdata(i).sessid = 1;
    %towers times
    rawdata(i).leftbups = score.trial(i).relCueOnset{1};
    rawdata(i).rightbups = score.trial(i).relCueOnset{2};
    rawdata(i).std = cell2mat({score.roi.noise});
    
    t = 0:dt:rawdata(i).T;
    rawdata(i).binned_leftbups = qfind(t, rawdata(i).leftbups);
    rawdata(i).binned_rightbups = qfind(t, rawdata(i).rightbups);
    
end

save(fullfile(path,'k56_1.mat'),'rawdata');