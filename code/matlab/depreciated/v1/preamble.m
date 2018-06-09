function preamble

%THIS WILL CREATE A NUMBER OF GLOBAL VARIABLES USED THROUGHOUT THE MODEL

global n dt nstdc dimy dimz dimd settle fr_func

%n: number of spatial bins
%nstdc: number of stds to use when convolving input pulses with the latent
    %variable distribution
%dt: temporal bin width
%dimz: number of parameters in the latent variable model
%dimd: number of parmaeters of choice observation distribution
%dimy: number of parameters of neural data distribution
%settle: force method to use "settling" method of Brunton, instead of
    %slightly faster convolution of pulses and finite-differencing scheme for
    %propogating the latent variable distribution

n = 203; nstdc = 6; dt = 2e-2; dimz = 8; dimd = 1; dimy = 4; settle = false;

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