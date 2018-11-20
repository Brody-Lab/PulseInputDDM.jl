using ForwardDiff:  Dual, partials

#value of the derivative should eventually become NaNs as x becomes very large (x will be inf, labmda will be 0 and derivative will become NaN)

xs = linspace(-800,800,10) #values of x to loop over
deriv = Array{Float64}(10,) #empty array
              
#loop over values of x
for i = 1:length(xs)
    x = Dual(xs[i],1.0) #make a dual variable to compute the gradient
    lambda = 1. / exp(x) #residual elements of a sigmoid that illustrate the problem
    deriv[i] = partials(lambda)[1] #gather the derivative component of the dual variable
end
