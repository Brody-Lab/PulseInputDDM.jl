function [rate,conds,cta] = psth_cta(data,dt,filt_sd)

global N

% if nargin < 2
%     posterior = [];
% end

tri = numel(data);

%loop over trials
for i = 1:tri
    
    %compute the cumulative diff of clicks
    t = 0:dt:data(i).nT*dt;
    ediffLR(i) = sum(-histcounts(data(i).leftbups,t) + histcounts(data(i).rightbups,t))/data(i).T;
    
end

% make clrs cell array if it isn't provided
n_conds = 3;
conds = bin_vals(ediffLR, n_conds, 'split_zero', 0);

T = cell2mat({data.T});
rate = NaN(N,ceil(max(T/dt)),tri);
%norm_post = NaN(N,ceil(max(T/dt)),tri);

%b = (1/3)* ones(1,3);

% if ~isempty(posterior)
%
%     switch dimy
%         case 4
%             ahat = x([1:N]+dimz); bhat = x([1:N]+dimz+N); chat = x([1:N]+dimz+2*N); dhat = x([1:N]+dimz+3*N);
%         case 3
%             ahat = x([1:N]+dimz); chat = x([1:N]+dimz+N); dhat = x([1:N]+dimz+2*N);
%         case 2
%             chat = x([1:N]+dimz); dhat = x([1:N]+dimz+N);
%     end
%
% end

for i = 1:n_conds
    for j = 1:tri
        if conds(j) == i
            
            %             if ~isempty(posterior)
            %                 blah = posterior{j}' * xc';
            %                 switch dimy
            %                     case 4
            %                         blah = fr_func([ahat,bhat,chat,dhat],blah');
            %                     case 3
            %                         blah = fr_func([ahat,chat,dhat],blah');
            %                     case 2
            %                         blah = fr_func([chat,dhat],blah');
            %                 end
            %                 norm_post(:,1:size(blah,1),j) = filter(b,1,blah)';
            %             end
            
            %            rate(data(j).N,1:size(data(j).spike_counts,1),j) = filter(b,1,data(j).spike_counts/dt)';
            for jj = 1:numel(data(j).N)
                try
                    rate(data(j).N(jj),1:size(data(j).spike_counts,1),j) = ...
                        FilterSpikes(filt_sd,data(j).spike_counts(:,jj)/dt)';
                catch
                    rate(data(j).N(jj),1:size(data(j).spike_counts,1),j) = data(j).spike_counts(:,jj)/dt';
                end
            end
        end
    end
end

%%

if nargout > 2
    
    
    Lcta = []; Rcta = [];
    
    for i = 1:tri
        
        %loop over left clicks for this trial
        for j = 2:numel(data(i).hereL)
            %add a new entry for this click
            Lcta = cat(3,Lcta,NaN(N,0.5/dt));
            %start of firing rate
            s = data(i).hereL(j);
            %end of firing rate
            e = s + 0.5/dt - 1;
            %go to end of trial if trial ends
            e = min([data(i).nT,e]);
            %collect spike counts during that window
            Lcta(:,1:numel(s:e),end) = rate(:,s:e,i);
        end
        
        %do same for right clicks
        for j = 2:numel(data(i).hereR)
            Rcta = cat(3,Rcta,NaN(N,0.5/dt));
            s = data(i).hereR(j);
            e = s + 0.5/dt - 1;
            e = min([data(i).nT,e]);
            Rcta(:,1:numel(s:e),end) = rate(:,s:e,i);
        end
        
    end
    
    %cat over left and right, and compute mean across clicks
    %cta = filter(b,1,nanmean(cat(3,Rcta,-Lcta),3)'/dt)';
    % cta = nanmean(cat(3,Rcta,-Lcta),3)/dt;
    %cta = nanmean(cat(3,Rcta,-Lcta),3);
    cta = cat(3,Rcta,-Lcta);
    % for i = 1:size(cta,1)
    %     cta(i,:) = FilterSpikes(1,cta(i,:))';
    % end
    
end
