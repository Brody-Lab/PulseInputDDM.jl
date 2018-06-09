function x = reparameterize(x0,dt,model_type,N,xv,fit_vec)

%create some new global variables which will be used by many functions

x = x0; %start with static parameters

if nargin > 4
    x(fit_vec) = xv; %populate it with those parameters that are being optimized
end

%% parameters for latent variable

%          vari       inatt          B    lambda       vara    vars     phi    tau_phi
%lb =      [1e-6       0              2    -1/(2*dt)    1e-6    1e-6     0.1    0.01]';
%ub =      [inf        1.0            inf   1/(2*dt)    inf     inf      inf    inf]';

x([1,5,6]) = x([1,5,6]).^2;
x(2) = 0.5*(1+tanh(x(2)));
x(3) = 2 + x(3)^2;
x(4) = -1/(2*dt) + (1/dt)*(0.5*(1+tanh(x(4))));
x(7) = 0.1 + x(7)^2;
x(8) = 0.02 + x(8)^2;

switch model_type
    
    case 'choice'
        
        x(9) = 0.5*(1 + tanh(x(9)));
        
    case 'spikes'
        
        x(8+1:8+2*N) = x(8+1:8+2*N).^2;
        
    case 'joint'
        
        x(9) = 0.5*(1 + tanh(x(9)));
        x(9+1:9+2*N) = x(9+1:9+2*N).^2;
        
end

end
