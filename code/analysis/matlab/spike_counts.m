%wrote this to look for cells that drop out

clear all;
ratnames = {'B053','B068','T011','T030','T034','T035','T036','T063','T068'};

files = dir('/Users/briandepasquale/Documents/Dropbox/hanks_data_sessions');

for i = 7
    
    for kk = 1:numel(files)
        
        if strfind(files(kk).name,ratnames{i})
            
            load(fullfile('/Users/briandepasquale/Documents/Dropbox/hanks_data_sessions',...
                files(kk).name));
            
            SC = NaN(size(rawdata(1).St,2),numel(rawdata));
            
            for j = 1:numel(rawdata)
                for k = 1:size(rawdata(1).St,2)
                    if size(rawdata(j).St{k},2) > 0
                        SC(k,j) = size(rawdata(j).St{k},1);
                    end
                end
            end
                        
            ns = find(~isnan(SC(:,1)));
            figure(kk);clf;hold on;
            for ii = 1:sum(~isnan(SC(:,1)))
                SC(ns(ii),:) = movmean(SC(ns(ii),:),50);
                subplot(2,ceil(sum(~isnan(SC(:,1)))/2),ii);hold on;
                plot(numel(rawdata)*ones(100,1),linspace(0,max(SC(ns(ii),:)),100),'g--');
                plot(SC(ns(ii),:));
            end
            title(files(kk).name(strfind(files(kk).name,'_')+1:strfind(files(kk).name,'.')-1));
        end
        
    end
    
end