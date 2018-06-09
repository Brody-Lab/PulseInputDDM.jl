function x = deparameterize(x0,model_type,N,xv,fit_vec)

%create some new global variables which will be used by many functions
global dt

x = x0; %start with static parameters

if nargin > 3
    x(fit_vec) = xv; %populate it with those parameters that are being optimized
end

%% parameters for latent variable

%          vari       inatt          B    lambda       vara    vars     phi    tau_phi
%lb =      [1e-6       0              2    -1/(2*dt)    1e-6    1e-6     0.1    0.01]';
%ub =      [inf        1.0            inf   1/(2*dt)    inf     inf      inf    inf]';

x([1,5,6]) = sqrt(x([1,5,6]));
x(2) = atanh(2*x(2)-1);
x(3) = sqrt(x(3)-2);
x(4) = atanh((2 * dt * (x(4) + 1/(2*dt))) - 1);
x(7) = sqrt(x(7) - 0.1);
x(8) = sqrt(x(8) - 0.02);

switch model_type
    
    case 'choice'
        
        x(9) = atanh(2*x(9) - 1);
        
    case 'spikes'
        
        x(8+1:8+2*N) = sqrt(x(8+1:8+2*N));
        
    case 'joint'
        
        x(9) = atanh(2*x(9) - 1);
        x(9+1:9+2*N) = sqrt(x(9+1:9+2*N));
        
end

end
