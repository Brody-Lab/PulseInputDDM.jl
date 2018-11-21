clearvars;

global N n tri dimz dimd use settle dt nstdc

ratnames = {'B053','B068','T011','T030','T034','T035','T036','T063','T068'};
ratnum = 4;
numrat = numel(ratnames);
iter = ratnum + 1*numrat;

save_pth = '~/Desktop';
in_pth = '~/Documents/latent_accum_models/hanks_data';

[data,ratname,model_type,~,lb,ub,~] = preamble(iter,in_pth,save_pth,[]);

%%

pth = '~/Documents/Dropbox/results';
if iter > 18
    pth2 = 'matlab_13106865';
else
    pth2 = 'matlab_13106656';    
end
x = load(fullfile(pth,pth2,sprintf('%s_%s.mat',ratname,model_type)),'xf');
x = x.xf;

%%

if iter > 9 && 1
    
    my_sig = @(x,y)y(1) + y(2) ./ (1 + exp(-y(3)*x + y(4)));
    
    figure;hold on;
    
    for i = 1:N
        xc = linspace(-x(3),x(3),100);
        plot(xc, my_sig(xc,x(8+i:N:end)));
    end
end

%%

LLstar = LL_all_trials(x,data,dt,n,tri,nstdc,dimz,dimd,use,N,settle);

%%

if 1
    
    fh = figure;set(fh,'color','w','Toolbar','none','Menubar','none');
    
    if iter > 9
        a = 4;
        b = 7;
    else
        a = 3;
        b = 3;
    end
    
    param_vec = [1:12];
    
    for param = param_vec
        
        dxs = linspace(lb(param),ub(param),20);
        
        LL = NaN(numel(dxs),1);
        
        for i = 1:numel(dxs)
            xx = x;
            xx(param) = dxs(i);
            LL(i) = LL_all_trials(xx,data,dt,n,tri,nstdc,dimz,dimd,use,N,settle);
        end
        
        fh;
        subplot(a,b,param);
        plot(dxs,LL);hold on;
        plot(x(param) * ones(100,1),linspace(min(LL),max(LL),100),'g--');
        drawnow;
        
    end
    
end

%%

fh = figure;set(fh,'color','w','Toolbar','none','Menubar','none');

a = 1; b = 2;

param_vec = [1,7];

for param = 1:numel(param_vec)
    
    dxs = linspace(lb(param_vec(param)),ub(param_vec(param)),40);
    
    LL = NaN(numel(dxs),1);
    
    for i = 1:numel(dxs)
        xx = x;
        xx(param_vec(param)) = dxs(i);
        LL(i) = LL_all_trials(xx,data,dt,n,tri,nstdc,dimz,dimd,use,N,settle);
    end
    
    fh;
    subplot(a,b,param);
    plot(dxs,LL);hold on;
    plot(x(param_vec(param)) * ones(100,1),linspace(min(LL),max(LL),100),'g--');
    drawnow;
    
end

%%

if 0
    
    xf = x;
    save(fullfile(save_pth,sprintf('%s_%s.mat',ratname,model_type)),'xf');
    
    system(sprintf('/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia -p auto ~/Documents/Dropbox/latent_accum/Code/julia/compute_Hessian_v2.jl %g %s %s', iter, save_pth, save_pth));
    load(fullfile(save_pth,sprintf('H_%s_%s.mat',ratname,model_type)),'H');
    julia_H = H;
    save(fullfile(save_pth,sprintf('%s_%s.mat',ratname,model_type)),'julia_H','-append');
    
end
