% XY = compute_proportion_ellipse(mu, Sigma, p)
%
%   Computes the isoline of the ellipse that contains P of the mass of a
%   two-dimension Gaussian distribution
%
%=PARAMETERS
%
%   mu
%       2-element numeric indicating the means of the Gaussan
%
%   Sigma2
%       2x2 numeric specifying the covariance matrix
%
%   p
%       scalar numeric indicating the proportation of the Gaussian
%       distribution to be specified by the isoline
%
%=OUTPUT
%   XY 
%       2 by 100 numeric whose top row specifies the X position and the
%       bottom specifies the Y position of the isoline

function XY = compute_proportion_ellipse(mu, Sigma2, p)
    validateattributes(mu, {'numeric'}, {'numel', 2})
    validateattributes(Sigma2, {'numeric'}, {'size', [2,2]})
    validateattributes(p, {'numeric'}, {'scalar'})
    mu = mu(:);
    [V, D] = eig(Sigma2);
    n_std = norminv((1+p)/2);
    t = linspace(0, 2 * pi);
    XY = n_std*(V * sqrt(D)) * [cos(t(:))'; sin(t(:))'] + mu;