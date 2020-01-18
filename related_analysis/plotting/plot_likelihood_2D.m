% plot_likelihood_2D(ML_vals, hessian)
%
%   Plots the isoline of an ellipse containing 95% of the loglikelihood
%   distribution for each pair of parameters
%
%=PARAMETERS
%
%   ML_vals
%       a numeric array of the maximum likelihood values of the parameters
%       of the fitted model. The number of elements in ML_VALS has to match
%       the number of rows and number of columns of the HESSIAN matrix
%
%   hessian
%       a numeric square array that indicates the local curvature of the
%       loglikelikehood function at maximum likelihood values of the model
%       parameters
%
%=OPTIONAL NAME-VALUE PAIRS
%
%   axes_size
%       two-element numeric indicating the size of the axes. The first
%       element is the width and the second the height. Default is [400,
%       300]
%
%   color
%       numeric, 1x3, each element between 0 an 1, indicating the [red,
%       green, and blue] intensity. Default is [0,0,0], black.
%
%   figure_handle
%       handle to the figure in which to create the axes. Useful for
%       overlaying ellipses from different conditions or animals on the
%       same plot for comparison. Default is empty, which creates a new
%       figure.
%
%   lower_bounds
%       constrain each axes to not be lower these bounds. Must have the
%       same number of elements as ML_VALS. Default is -infinity.
%
%   log_scale
%       An numeric, char, string, or cell array indicating the parameters
%       that should be plotted in log scale. If numeric values are
%       supplied, than they are interpreted to be indices of the ML_vals.
%       Otherwise, they are intepreted to be names of the parameters
%       supplied in PARAM_NAMES. Default is [], indicating all parameters
%       are plotted on a linear scale.
%
%   marker
%       specifier for indicating the maximum likelihood values
%
%   
%
%   upper_bounds
%       constrain each axes to not be larger these bounds. Must have the
%       same number of elements as ML_VALS. Default is -infinity.
%
%=OPTIONAL POSITIONAL OUTPUTS
%   
%   1) figure_handle
%       handle to the figure
%
%   2) P
%       A structure listing the parameters used for plotting
%
%=EXAMPLE:
%
%   Plots the 95% confidence intervals from two conditions in the same
%   plot:
%
%       >>fig_hdl = plot_likelihood_2D(ML_params{1}, hessian{1}); 
%       >>plot_likelihood_2D(ML_params{2}, hessian{2}, 'figure_handle', fig_hdl); 
function varargout = plot_likelihood_2D(ML_vals, hessian, varargin)
if numel(ML_vals) ~= size(hessian,1) || ...
   numel(ML_vals) ~= size(hessian,2)
    error(['The number of elements in ML_VALS has to match ' ...
           'the number of rows and number of columns of HESSIAN'])
end
P = inputParser;
addParameter(P, 'axes_size', [400 300], @(x) validateattributes(x, {'numeric'}, {'numel', 2}))
addParameter(P, 'color', [0,0,0], @(x) validateattributes(x, {'numeric'}, {'size', [1,3], '>=' 0, '<=', 1}))
addParameter(P, 'lower_bounds', -inf*numel(ML_vals), ...
    @(x) validateattributes(x, {'numeric'}, {'numel', numel(ML_vals)}))
addParameter(P, 'figure_handle', [], @(x) ishandle(x) && isscalar(x))
addParameter(P, 'log_scale', [], ...
    @(x) validateattributes(x, {'numeric', 'char', 'string', 'cell'}, {}))
addParameter(P, 'marker', '+', @(x) ischar(x))
addParameter(P, 'param_names', '', @(x) ischar(x) || isstring(x) || iscell(x))
addParameter(P, 'upper_bounds', inf*numel(ML_vals), ...
    @(x) validateattributes(x, {'numeric'}, {'numel', numel(ML_vals)}))
parse(P, varargin{:});
P = P.Results;
n_params = numel(ML_vals);
if ischar(P.param_names) || iscell(P.param_names)
    P.param_names = string(P.param_names);
end
if ~isempty(P.figure_handle)
    P.figure_handle = figure(P.figure_handle);
else
    P.figure_handle = figure('Position', [0,0, n_params*P.axes_size(1), ...
                                               n_params*P.axes_size(2)]);
end
S2 = hessian^-1; % covariance matrix
for i = 1:n_params
    for j = i+1:n_params
        subplot_ind = (i-1)*n_params +j;
        subplot(n_params, n_params, subplot_ind);
        fig_prepare_axes
        set(gca, 'XLimMode', 'auto', 'YLimMode', 'auto')
        mu = [ML_vals(i); ML_vals(j)];
        s2 = S2([i,j], [i,j]);
        if all(diag(s2)>0)
            XY = compute_proportion_ellipse(mu, s2, 0.95);
            plot(XY(1, :), XY(2, :), 'Color', P.color);
        end
        plot(ML_vals(i), ML_vals(j), P.marker, 'Color', P.color);
        if ~isempty(P.param_names)
            ylabel(P.param_names{j})
            xlabel(P.param_names{i})
        end
        x_lim = xlim;
        y_lim = ylim;
        if x_lim(1)<P.lower_bounds(i)
            x_im(1)=P.lower_bounds(i);
        end
        if x_lim(2)>P.upper_bounds(i)
            x_im(2)=P.upper_bounds(i);
        end
        if y_lim(1)<P.lower_bounds(j)
            y_lim(1)=P.lower_bounds(j);
        end
        if y_lim(2)>P.upper_bounds(j)
           y_lim(2)=P.upper_bounds(j);
        end
        set(gca, 'XLim', x_lim, 'YLim', y_lim)
        if isnumeric(P.log_scale)
            if any(P.log_scale == i)
                set(gca, 'XScale', 'log')
            end
            if any(P.log_scale == j)
                set(gca, 'YScale', 'log')
            end
        else
            if any(cellfun(@(x) contains(P.param_names{i},x), P.log_scale))
                set(gca, 'XScale', 'log')
            end
            if any(cellfun(@(x) contains(P.param_names{j},x), P.log_scale))
                set(gca, 'YScale', 'log')
            end
        end
    end
end
if nargout > 0
    varargout{1} = P.figure_handle;
end
if nargout > 1
    varargout{2} = P;
end