using ForwardDiff:  Dual, partials

n = 203
N = 1
a = Dual(0,0.0)
b = Dual(6,0.0)
B = Dual(10,1.0)
d = Dual(40,0.0)
c = Dual(75,0.0)
#d = Dual(-18,0.0)
#c = Dual(-56,0.0)
dx = 2.*B/(n-2); 
xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2)));
              
lambda = Array{Dual}(n,N)
temp = broadcast(+,broadcast(*,-c',xc),d')
lambda[(exp.(temp) .< Inf) .& (exp.(temp) .> 0.0)] = broadcast(+, a', broadcast(/, b', (1 + broadcast(exp, broadcast(+, broadcast(*, -c', xc), d')))))[(exp.(temp) .< Inf) .& (exp.(temp) .> 0.0)]
lambda[exp.(temp) .== 0.0] = broadcast(+,a',broadcast(/,b',ones(n,)))[exp.(temp) .== 0.0]
lambda[exp.(temp) .== Inf] = broadcast(*,a',ones(n,))[exp.(temp) .== Inf]
LL = sum(lambda)
