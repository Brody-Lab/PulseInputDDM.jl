function g = lambda_grads

clearvars;

cs = [66.20,66.25];
B = 10;
d = 40;

g = zeros(3,length(cs));
dx = 1e-6;

for i = 1:length(cs)
        
    for j = 1:3
        
        xf = [B,cs(i),d];
        
        xf2 = xf;
        xf2(j) = xf2(j) - dx/2;
        LL1 = test(xf2);
        
        xf2 = xf;
        xf2(j) = xf2(j) + dx/2;
        LL2 = test(xf2);
        
        g(j,i) = (LL2 - LL1)/dx;
        
    end
    
end

disp('done');

end

function LL = test(x)

n = 203;

B = x(1);
c = x(2);
d = x(3);

dx = 2*B/(n-2); %spatial bin width of latent variable
xc = [linspace(-(B+dx/2),-dx,(n-1)/2),0,linspace(dx,(B+dx/2),(n-1)/2)]; %location of bin centers
lambda = 1./ (1 + exp(-c * xc + d));

LL = sum(lambda);


end