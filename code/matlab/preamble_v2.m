function preamble_v2

%THIS WILL CREATE A NUMBER OF GLOBAL VARIABLES USED THROUGHOUT THE MODEL

global n dt dimy dimz dimd fr_func

%n: number of spatial bins
%dt: temporal bin width
%dimz: number of parameters in the latent variable model
%dimd: number of parmaeters of choice observation distribution
%dimy: number of parameters of neural data distribution

n = 203; dt = 2e-2; dimz = 8; dimd = 1; dimy = 4;

%choose a firing rate function
switch dimy
    %sigmoid
    case 4
        fr_func = @(x,z)bsxfun(@plus,x(:,1),bsxfun(@rdivide,x(:,2),...
            (1 + exp(bsxfun(@plus,-x(:,3) * z,x(:,4))))))' + eps;
    %softplus
    case 3
        fr_func = @(x,z)bsxfun(@plus,x(:,1),log(1 + exp(bsxfun(@plus,-x(:,2) * z,x(:,3)))))' + eps;
    %exponential
    case 2
        fr_func = @(x,z)exp(bsxfun(@plus,-x(:,1) * z,x(:,2)))' + eps;
end

end