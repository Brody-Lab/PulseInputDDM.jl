using ForwardDiff:  Dual, partials, gradient, value

x = [Dual(Dual(10.,1.,0.,0.),Dual(1.,0.,0.,0.),Dual(0.,0.,0.,0.,),Dual(0.,0.,0.,0.,)),Dual(Dual(-76.,0.,1.,0.),Dual(0.,0.,0.,0.,),Dual(1.,0.,0.,0.,),Dual(0.,0.,0.,0.)),Dual(Dual(-43.,0.,0.,1.),Dual(0.,0.,0.,0.),Dual(0.,0.,0.,0.),Dual(1.,0.,0.,0.))];

#x = [Dual(10.048090153087246,1.,0.,0.),Dual(-76.1177158435867,0.,1.,0.),Dual(-43.01,0.,0.,1.)]

x = [Dual(10.04,1.,0.,0.),Dual(-76.11,0.,1.,0.),Dual(-43.01,0.,0.,1.)]

x = [10.048090153087246,-76.1177158435867,-43.01]

function LL_func(x)
    
    n = 203
    N = 1
    B = x[1]
    a = 0.37;
    b = 6.37
    c = x[2]
    d = x[3]

    dx = 2.*B/(n-2); 
    xc = vcat(collect(linspace(-(B+dx/2.),-dx,(n-1)/2.)),0.,collect(linspace(dx,(B+dx/2.),(n-1)/2)));
    temp = broadcast(+,broadcast(*,-c',xc),d')

    #lambda = broadcast(*,a',ones(n,))[exp.(temp) .== Inf]
    #lambda = vcat(lambda,broadcast(+, a', broadcast(/, b', (1 + broadcast(exp, broadcast(+, broadcast(*, -c', xc), d')))))[(exp.(temp) .< Inf) .& (exp.(temp) .> 0.0)])
    #lambda = vcat(lambda,broadcast(+,a',broadcast(/,b',ones(n,)))[exp.(temp) .== 0.0])

    #lambda = Array{Dual}(n,N)
    temp = broadcast(+,broadcast(*,-c',xc),d')
    lambda = broadcast(+, a', broadcast(/, b', (1 + broadcast(exp, broadcast(+, broadcast(*, -c', xc), d')))))
    #lambda[exp.(temp) .== 0.0] = broadcast(+,a',broadcast(/,b',ones(n,)))[exp.(temp) .== 0.0]
    #lambda[exp.(temp) .== Inf] = broadcast(*,a',ones(n,))[exp.(temp) .== Inf]
    lambda[exp.(temp) .<= 1e-150] = broadcast(+,a',broadcast(/,b',ones(n,)))[exp.(temp) .<= 1e-150]
    lambda[exp.(temp) .>= 1e150] = broadcast(*,a',ones(n,))[exp.(temp) .>= 1e150]
    LL = sum(lambda)

    return LL

end
