using LinearAlgebra, SparseArrays, Random, Distributions, Plots, StatsBase

# Clear all variables
T = 1.0          # time of simulation
dt = 2e-2        # Fokker-Planck approximations timestep
nMC = 1e4        # number of Monte Carlo particles
dtMC = 1e-4      # Monte Carlo timestep

B = 1.0          # bound height
vara = 1e-2      # diffusion variance
lambda = -1e-3   # drift
mu0 = 0.0 * B    # initial mean of distribution
n = 53           # number of bins

dx = 2 * B / (n - 2)   # bin width
dx2 = dx^2             # dx^2
vari = dx2             # make initial distribution as large as dx^2

# Finite difference

xe = vcat(-(B+dx), range(-B, -dx/2, ceil(Int, (n-3)/2+1))..., 
          range(dx/2, B, floor(Int, (n-3)/2+1))..., B+dx) # edges of the bins
xc = (xe[1:n] .+ dx/2)'  # bin centers

cDiff = vara / (dx^2 * 2)  # scale factor for diffusion
cDrft = lambda / (dx * 2)  # scale factor for drift

# Diffusion matrix
Ddff = diagm(0 => -2 * cDiff * [0; ones(n-2); 0]) +
       diagm(-1 => cDiff * [0; ones(n-2)]) +
       diagm(1 => cDiff * [ones(n-2); 0])

# Drift matrix
Dder = diagm(-1 => -cDrft * [0; ones(n-2)]) +
       diagm(1 => cDrft * [ones(n-2); 0])

# Multiply by a values and add
DD = -Dder * diagm(0 => xc') + Ddff

# Matrix exponential
M = exp(DD * dt)

# Brunton method

# You would replace this with your implementation of `make_F` in Julia
M_B = PulseInputDDM.transition_M(vara*dt, lambda, 0., dx, xc', n, dt);

# Initialize

Pa = pdf.(Normal(mu0, sqrt(vari)), xc) .* dx
Pa /= sum(Pa)  # Fin. Diff.
Pa = collect(Pa')
Pa_B = copy(Pa)  # Brunton

# For saving
PA = zeros(n, round(Int, T/dt))
PA_B = similar(PA)

# Propagate

for t in 1:round(Int, T/dt)
    global Pa, Pa_B  # Ensure these refer to the global variables
    PA[:, t] = Pa
    PA_B[:, t] = Pa_B
    
    Pa = M * Pa  # Fin. Diff.
    Pa_B = M_B * Pa_B  # Brunton (Uncomment this once you define `M_B`)
    
end

# Monte Carlo

# For saving
PMC = fill(NaN, size(PA))

a = mu0 .+ sqrt(vari) .* randn(Int(nMC))  # Initialize Gaussian

for t in 1:round(Int, T/dtMC)

    global a

    a[a .< -(B+dx)] .= -(B+dx/2)
    a[a .> B+dx] .= B+dx/2
    
    # Bin the individual particles
    if mod(t-1, round(Int, dt/dtMC)) + 1 == 1
        # Bin the particles and normalize by the number of Monte Carlo particles
        counts = StatsBase.fit(Histogram, a, xe).weights
        PMC[:, ceil(Int, dtMC*(t)/dt)] = (1/nMC) * counts
        #PMC[:, ceil(Int, dtMC*(t)/dt)] = (1/nMC) * histcounts(a, xe)
    end
    
    go = (a .< B) .& (a .> -B)  # Only integrate those that haven't crossed the boundary
    
    # OU process: 2 terms, 1-drift and 2-diffusion
    a[go] .= a[go] .+ (lambda * dtMC) .* a[go] .+ sqrt(vara * dtMC) .* randn(sum(go))
end

# Plot over time

p = plot()
plot!(p, xc', PA[:, 1], label="Fin.Diff.", color="red", lw=2, marker=:o)
plot!(p, xc', PA_B[:, 1], label="Brunton", color="green", lw=2, marker=:star)
plot!(p, xc', PMC[:, 1], label="Monte Carlo", color="blue", lw=2, marker=:star)
display(p)

for i in 1:size(PA, 2)
    p = plot()
    plot!(p, xc', PA[:, i], label="Fin.Diff.")
    plot!(p, xc', PA_B[:, i], label="Brunton")
    plot!(p, xc', PMC[:, i], label="Monte Carlo")
    sleep(0.1)
    display(p)
end
