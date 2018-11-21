fr_func = @(x,z)bsxfun(@plus,x(:,1),bsxfun(@rdivide,x(:,2),...
    (1 + exp(bsxfun(@plus,-x(:,3) * z,x(:,4))))))' + eps;

figure; hold on;

bs = [1,10,100];

for i =1:numel(bs)
    x = linspace(-10,10,100); param = [0,bs(i),1,0];    
    plot(x,fr_func(param,x))
end
