% Compute the net input (right-left) and total input (right+left) of
% clicks for all time steps.  
%
% OUTPUTS
% --------
% net_input: array of length N with net input of clicks at each time step
% (right, weighted clicks - left, weighted clicks at each time step)
%
% tot_input: array of length N with total input of clicks at each time step
% (right, weighted clicks + left, weighted clicks at each time step)
%
% INPUTS
% ---------
% t: time axis
% leftbups: times of left clicks
% rightbups: times of right clicks
% clicks_L: adapted weights of left clicks
% clicks_R: adapted weights of right clicks

function [net_input, tot_input] = click_inputs(t, leftbups, rightbups, clicks_L, clicks_R)

here_L = qfind(t, leftbups);
here_R = qfind(t, rightbups);

net_input = zeros(length(t),1);
tot_input = zeros(length(t),1);

for i = 1:numel(leftbups)
    net_input(here_L(i)) = net_input(here_L(i)) - clicks_L(i);
    tot_input(here_L(i)) = tot_input(here_L(i)) + clicks_L(i);
end

for i = 1:numel(rightbups)
    net_input(here_R(i)) = net_input(here_R(i)) + clicks_R(i);
    tot_input(here_R(i)) = tot_input(here_R(i)) + clicks_R(i);
end
