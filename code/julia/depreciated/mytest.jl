
using ForwardDiff: gradient, Dual

n = 203 #number of spatial bins
B = 10 #bound height
cs = [66.20,66.25,66.5,66.75] 
#cs = linspace(66.22891,66.22898,100) #various values of the sigmoid gain
d = 40 #sigmoid offset
g = zeros(3,length(cs)) #empty array for autodiff grad
g2 = zeros(g) #empty array for numerical grad
dx = 1e-6 #step size for numerical grad

function comp_sig{TT}(x::Vector{TT},n)

    #remaining compnents of my original LL function, but which still results in grad NaNs
    #This function will set the location of the bin centers, define a sigmoid function at
    #those points, and sum that function
    B,c,d = x;
   
    dx = 2.*B/(n-2); #bin width
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2))); #bin centers

    lambda = 1. ./ (1. + exp.(-c*xc + d)) #sigmoid
    LL = sum(lambda) #sum
        
    return LL

end

#loop over sigmoid gain values. At a point, gradient with become NaNs
for i = 1:length(cs)

    x0 = [B,cs[i],d]
    #compute autograd
    g[:,i] = gradient(x -> comp_sig(x,n), x0)

    #loop over parameters and compute gradient for each one numerically
    for j = 1:3
        x1 = copy(x0);
        x1[j] = x1[j] - dx/2.
        LL1 = comp_sig(x1,n)
        x1 = copy(x0)
        x1[j] = x1[j] + dx/2.
        LL2 = comp_sig(x1,n)
        g2[j,i] = (LL2 - LL1)/dx

    end
end
