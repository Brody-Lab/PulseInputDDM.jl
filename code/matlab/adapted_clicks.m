function [L,R] = adapted_clicks(leftbups, rightbups, phi, tau_phi)
%% Defaults
if nargin < 3 || isempty(phi)
    phi = 1;
end

if nargin < 4 || isempty(tau_phi)
    tau_phi = 0;
end

%% Main
% compute the effectives sizes of each click in leftbups and rightbups
% considering the sensory adaptation parameters

% returns L and R, the same sizes as leftbups and rightbups.
% if there's no appreciable adaptation, returns all ones

L = ones(size(leftbups));  R = ones(size(rightbups)); 

% magnitude of stereo clicks set to zero
if ~isempty(leftbups) && ~isempty(rightbups) && abs(leftbups(1)-rightbups(1)) < eps
    L(1) = eps;
    R(1) = eps;
end

% if there's appreciable same-side adaptation
if phi ~= 1

% inter-click-intervals
ici_L = diff(leftbups);
ici_R = diff(rightbups);

for i = 2:numel(leftbups)
    if (1-L(i-1)*phi) <= 1e-150
        L(i) = 1;
    else
        last_L = tau_phi*log(1-L(i-1)*phi);
        L(i) = 1 - exp((-ici_L(i-1) + last_L)/tau_phi);
    end
end

for i = 2:numel(rightbups)
    if (1-R(i-1)*phi) <= 1e-150
        R(i) = 1;
    else
        last_R = tau_phi*log(1-R(i-1)*phi);
        R(i) = 1 - exp((-ici_R(i-1) + last_R)/tau_phi);
    end
end

L = real(L); R = real(R);

end
