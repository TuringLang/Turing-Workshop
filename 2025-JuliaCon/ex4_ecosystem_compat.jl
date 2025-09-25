using Turing, DifferentialEquations

# Define the differential equation to solve
# du/dt = p sin(t) u(t)
function diffeq(du, u, p, t)
   du[1] = p*sin(t)*u[1]
   return nothing
end

# Define initial-value problem.
prob = ODEProblem(diffeq, [1.0], (0.0, 10.0), 0.5)

@model function fit_diffeq(data, prob)
   # Priors
   p ~ truncated(Exponential(1.0); upper=100.0)
   # Solve the ODE
   predicted = solve(prob, Tsit5(); p=p, saveat=0.1, abstol=1e-6, reltol=1e-6)
   # Observations
   data ~ product_distribution(Poisson.(vcat(predicted.u...) .+ 1e-5))
   return nothing
end

# Make some synthetic data
odedata = vcat(solve(prob, Tsit5(); p=0.5, saveat=0.1, abstol=1e-6, reltol=1e-6).u...)
odedata = Int.(round.(odedata + randn(length(odedata)) .* 0.1))

# Create the model and sample 3 independent chains
model = fit_diffeq(odedata, prob)
chain = sample(model, NUTS(), MCMCSerial(), 10_000, 3)
describe(chain)
