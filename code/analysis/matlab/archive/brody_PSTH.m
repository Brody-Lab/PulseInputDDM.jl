function [norm_ys2,conds,clrs] = brody_PSTH(data,posterior,xc,x)

global dt dimy dimz fr_func N

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

conds = bin_vals(ediffLR, 4, 'split_zero', 1);

% make clrs cell array if it isn't provided
n_conds = 4;
max_br = 1;
min_br = 0.4;
n_each_clr = floor(n_conds/2);
br_vec = linspace(min_br,max_br,n_each_clr);
for i=1:n_conds
    if i <= n_each_clr
        % reds for low conditions
        clrs(i,:) = [br_vec(i) 0 0];
    elseif i > ((n_conds+1)/2)
        % greens for high conditions
        clrs(i,:) = [0 br_vec(end-i+n_conds-n_each_clr+1)  0];
    else
        % black for middle condition is there is one
        clrs(i,:) = [0.7 0.7 0];
    end
end

T = cell2mat({data.T});
norm_ys2 = NaN(N,ceil(max(T/dt)),tri);
%norm_post = NaN(N,ceil(max(T/dt)),tri);

b = (1/3)* ones(1,3);

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
            
            norm_ys2(data(j).N,1:size(data(j).spike_counts,1),j) = filter(b,1,data(j).spike_counts/dt)';         
        end
    end
end