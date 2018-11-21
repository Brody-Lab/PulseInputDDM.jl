function H = num_Hessian(fun, x, dx)

n = numel(x);
H = zeros(n,n);
f0 = feval(fun, x);

if isscalar(dx)
    vdx = dx*ones(n,1);
elseif numel(dx) == n
    vdx = dx(:);
end

A = diag(vdx/2);

for j = 1:n
    
    dxs = [2,-2];
    f = NaN(2,1);
    
    for k = 1:2
        
        %central differences
        xdx = x + dxs(k)*A(:,j);
        
        f(k) = feval(fun, xdx);
        
    end

    H(j,j) = f(1)+f(2)-2*f0;
        
end
    
for j = 2:n
    for i = j+1:n
        
        f = NaN(4,1);
        dxs = [1,1; -1,-1; 1,-1; -1,1];
        
        for k = 1:4
            
            %central differences
            xdx = x + dxs(k,1)*A(:,j) + dxs(k,2)*A(:,i);
            
            f(k) = feval(fun, xdx);
            
        end
        
        H(j,i) = f(1)+f(2)-f(3)-f(4);
        H(i,j) = H(j,i);
        
    end
end

H = H./(vdx * vdx');
