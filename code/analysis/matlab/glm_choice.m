function glm_choice(ratname,model_type,sessid,in_pth,save_pth)

sessid = strsplit(sessid,'_'); %turn string of sessions, where each

%load everything from matlab run except xf and history
load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'-regexp', '^(?!(xf|history)$).');

try %to load xf from julia run, if exists
    load(fullfile(save_pth,sprintf('julia_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    
catch   %load xf and history from matlab run
    load(fullfile(save_pth,sprintf('%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)),'xf','history');
    try %to load Hessian after matlab run,if it exists
        load(fullfile(save_pth,sprintf('H_%s_%s_%s.mat',ratname,strjoin(sessid,'_'),model_type)));
    end
    
end

[data] = load_data(ratname,model_type,sessid,in_pth,dt);
%compute posterior
[posterior,xc] = LL_all_trials_v2(xf,data,dt,n,dimz,dimd,dimy,...
    use,N,fr_func,true,x0(~fit_vec),fit_vec);

%%

% tri = numel(data);
% LR = NaN(tri,1);
% 
% %loop over trials
% for i = 1:tri
%     
%     %compute the cumulative diff of clicks
%     t = 0:dt:data(i).nT*dt;
%     diffLR = reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
%         histcounts(data(i).rightbups,t)),[],1);
%     LR(i) = diffLR(end);
%             
% end
% 
% %%
% 
% ntrials = size(posterior,1);
% post = NaN(ntrials,1);
% LR = NaN(ntrials,1);
% choice = NaN(ntrials,1);
% 
% for i = 1:ntrials
%     post(i) = posterior{i}(:,end)' * xc';
%     choice(i) = data(i).pokedR;
%     t = 0:dt:data(i).nT*dt;
%     diffLR = reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
%         histcounts(data(i).rightbups,t)),[],1)';
%     LR(i) = diffLR(end);
% end
% 
% [bpost,devpost] = glmfit(post,choice,'binomial','link','logit');
% [bLR,devLR] = glmfit(LR,choice,'binomial','link','logit');
% %[bSC,devSC] = glmfit([SC],choice,'binomial','link','logit');
% %[bLR,devLR] = glmfit(diffLR,choice,'binomial','link','logit');
% 
% %xSCLR = bSCLR(2:end)'*[SC,LR]' + bSCLR(1);
% %xSC = -bSC(2:end)'*SC' - bSC(1);
% %xLR = -bLR(2:end)'*LR' - bLR(1);
% xpost = bpost(2:end)'*post'+ bpost(1);
% 
% xLR = bLR(2:end)'*LR'+ bLR(1);
% 
% %xpost = -bpost(2:end)'*post' - bpost(1);
% yhatpost = 1./(1+exp(xpost));
% yhatdiffLR = 1./(1+exp(xLR));
% 
% figure;scatter(xLR,yhatdiffLR);
% hold on;
% scatter(xpost,yhatpost);
%yhatpost = 1./(1+exp(xpost));
%yhatSC = 1./(1+exp(xSC));
%yhatLR = 1./(1+exp(xLR));
%yhatSC = 1./(1+exp(xSC));

%%

ntrials = size(posterior,1);
post = NaN(ntrials,n);
diffLR = NaN(ntrials,max(cell2mat({data.nT})));
choice = NaN(ntrials,1);

for i = 1:ntrials
    post(i,:) = posterior{i}(:,end);
    choice(i) = data(i).pokedR;
    t = 0:dt:data(i).nT*dt;
    diffLR(i,1:data(i).nT) = reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
        histcounts(data(i).rightbups,t)),[],1)';
end

%blah = post(choice == 1,:);

[bpost,devpost] = glmfit(post,choice,'binomial','link','logit');
[bdiffLR,devdiffLR] = glmfit(diffLR,choice,'binomial','link','logit');
%[bSC,devSC] = glmfit([SC],choice,'binomial','link','logit');
%[bLR,devLR] = glmfit(diffLR,choice,'binomial','link','logit');

%xSCLR = bSCLR(2:end)'*[SC,LR]' + bSCLR(1);
%xSC = -bSC(2:end)'*SC' - bSC(1);
%xLR = -bLR(2:end)'*LR' - bLR(1);
post2 = post;
post2(isnan(post2)) = 0;
xpost = bpost(2:end)'*post2'+ bpost(1);

diffLR2 = diffLR;
diffLR2(isnan(diffLR2)) = 0;
xdiffLR = bdiffLR(2:end)'*diffLR2'+ bdiffLR(1);

%xpost = -bpost(2:end)'*post' - bpost(1);
yhatpost = 1./(1+exp(xpost));
yhatdiffLR = 1./(1+exp(xdiffLR));

figure;scatter(xdiffLR,yhatdiffLR);
hold on;
scatter(xpost,yhatpost);
%yhatpost = 1./(1+exp(xpost));
%yhatSC = 1./(1+exp(xSC));
%yhatLR = 1./(1+exp(xLR));
%yhatSC = 1./(1+exp(xSC));

%%

% pm = NaN(size(posterior,1),max(cell2mat({data.nT})));
% choice = NaN(size(posterior,1),1);
% 
% for i = 1:size(posterior,1)
%     pm(i,1:data(i).nT) = posterior{i}'*xc';
%     choice(i) = data(i).pokedR;
% end

% post = NaN(size(posterior,1),max(cell2mat({data.nT}))*n);
% choice = NaN(size(posterior,1),1);
% 
% for i = 1:size(posterior,1)
%     post(i,1:data(i).nT*n) = posterior{i}(:);
%     choice(i) = data(i).pokedR;
% end
% 
% [b,dev,stats] = glmfit(post,choice,'binomial','link','logit');

% %%
% 
% post = NaN(size(posterior,1),n);
% choice = NaN(size(posterior,1),1);
% 
% for i = 1:size(posterior,1)
%     post(i,:) = posterior{i}(:,end);
%     choice(i) = data(i).pokedR;
% end
% 
% [b,dev,stats] = glmfit(post,choice,'binomial','link','logit');
% 
% blah = 1./(1+exp(-b(2:end)'*post' - b(1)));

%%

ntrials = 410;
postm = NaN(ntrials,1);
choice = NaN(ntrials,1);
%post = NaN(size(posterior,1),n);
post = NaN(size(posterior,1),max(cell2mat({data.nT})));
SC = NaN(ntrials,3);
LR = NaN(ntrials,1);

for i = 1:ntrials
    postm(i,1) = posterior{i}(:,end)' * xc';
    %postm(i,1) = mean(posterior{i}' * xc');
    choice(i) = data(i).pokedR;
    %post(i,:) = posterior{i}(:,end);
    SC(i,data(i).N) = sum(data(i).spike_counts);
    
    t = 0:dt:data(i).nT*dt;
    diffLR = reshape(cumsum(-histcounts(data(i).leftbups,t) + ...
        histcounts(data(i).rightbups,t)),[],1);
    LR(i) = diffLR(end);
    
end

[bpostm,devpostm] = glmfit(postm,choice,'binomial','link','logit');
%[bLR,devLR] = glmfit(LR,choice,'binomial','link','logit');
%[bpost,devpost] = glmfit(post,choice,'binomial','link','logit');
%[bSCLR,devSCLR] = glmfit([SC,LR],choice,'binomial','link','logit');
[bSC,devSC] = glmfit([SC],choice,'binomial','link','logit');
[bLR,devLR] = glmfit([LR],choice,'binomial','link','logit');

%xSCLR = bSCLR(2:end)'*[SC,LR]' + bSCLR(1);
xSC = -bSC(2:end)'*SC' - bSC(1);
xLR = -bLR(2:end)'*LR' - bLR(1);
xpostm = bpostm(2:end)'*postm'+ bpostm(1);
%xpost = -bpost(2:end)'*post' - bpost(1);
yhatpostm = 1./(1+exp(xpostm));
%yhatpost = 1./(1+exp(xpost));
yhatSC = 1./(1+exp(xSC));
yhatLR = 1./(1+exp(xLR));
%yhatSC = 1./(1+exp(xSC));

%%

figure;scatter(xLR,yhatLR);
hold on;
scatter(xpostm,yhatpostm);
scatter(xSC,yhatSC);
%scatter(xpost,yhatpost);

%%

% pred = 1./(1+exp(b(1)+b(2:end)'*pm2'));
% pred(pred<0.5) = 0;
% pred(pred>0.5) = 1;
% 
% figure;scatter(pred,choice);

end