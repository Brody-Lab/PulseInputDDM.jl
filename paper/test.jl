using LinearAlgebra, PyPlot

# Parameters
dt = 1e-2
B = 10.
λ = -4.
σ2 = 10.
n = 53

xc, dx = PulseInputDDM.bins(B, n);
P0 = PulseInputDDM.P0(dx^2, n, dx, xc, dt) 
M = PulseInputDDM.transition_M(σ2*dt, λ, 0., dx, xc, n, dt);

# Calculate coefficients
cDiff = σ2 * dt / (2*dx^2)
cDrft = λ * dt / (2 * dx)
Ddff = diagm(0 => ones(n) * -2cDiff, 1 => ones(n-1) * cDiff, -1 => ones(n-1) * cDiff)
Ddff[:, [1, n]] .= 0
Dder = diagm(0 => zeros(n), 1 => ones(n-1) * cDrft, -1 => ones(n-1) * -cDrft)
Dder[:, [1, n]] .= 0
C = exp(-Dder * diagm(0 => xc) + Ddff)

#plot(xc, P0, label="P0")
#plot(xc, M^(1/dt) * P0, label="M")
#plot(xc, C^(1/dt) * P0, label="C")
#legend()

plot(sort(real(eigvals(M))), label="realM")
plot(sort(abs.(imag(eigvals(M)))), label="imagM")
plot(sort(real(eigvals(C))),  label="realC")
plot(sort(abs.(imag(eigvals(C)))), label="imagC")
legend()