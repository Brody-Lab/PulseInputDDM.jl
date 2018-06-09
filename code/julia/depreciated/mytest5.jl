using ForwardDiff:  Dual, partials, value

#this function illustrates the undesirable behavior forward diff will give for the derivative of 1/(large number)
function obj1(x)
    y =  1./exp.(x) 
    return y
end

function obj2(x)
    y = x[1]^2 * x[2]^3
end

#######################   below this illustrates some undesirable properties that forwarddiff might produce when dealing with large numbers   ################

x1 = Dual(700.,1.) #this defines a dual variable with a value of 700. The "1." in the second location means we'd like to compute the first dervative
x2 = Dual(750.,1.)

y1 = obj1(x1) #gradient here should be very close to zero
y2 = obj1(x2) #now, gradient is a NaN (because x is larger)


#########################   below illustrates how to create a dual variable for directly computing the Hessian (for example) ####################

x3 = [Dual(Dual(3.,1.,0.),Dual(1.,0.,0.),Dual(0.,0.,0.)),Dual(Dual(2.,0.,1.),Dual(0.,0.,0.),Dual(1.,0.,0.))] #now we define a 2 dimensional dual with a value of [2.,1.]
#We "nested" duals and fill it out appropriately to compute the first derivative of each entry and the second derivatives (mixed too) for both variables.

y3 = obj2(x3) #now pass through a multi-dimensional function

v3 = value(y3) #this function will collect the value of the function and the first derivatives, which wil be the second values in the dual variable
#value should be 3^2 * 2^3 = 72. 
v3_1 = value(value(x3[1]))^2 * value(value(x3[2]))^3

#gradients should be as follows: g_1: 2*x[1]*x[2]^3 = 48., g_2: x[1]^2*3*x[2]^2 = 108 
g1 = 2*value(value(x3[1]))*value(value(x3[2]))^3
g2 = value(value(x3[1]))^2*3*value(value(x3[2]))^2

p3 = partials(y3) #this function will collect the second derivatives (actually, it looks like it collects the first order derivatives too, but I don't quit understand this)
#H_11 = 2*x[2]^3 = 16, H_12 = 2*x[1]*3*x[2]^2 = 72 =  H_21, H_22 = x[1]^2*3*2*x[3] = 108
H_11 = 2*value(value(x3[2]))^3
H_12 = 2*value(value(x3[1]))*3*value(value(x3[2]))^2
H_22 = value(value(x3[1]))^2*3*2*value(value(x3[2]))

