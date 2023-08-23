using DifferentialEquations
using DiffEqNoiseProcess
using Distributions
using LogDensityProblems, LogDensityProblemsAD
using UnPack

struct DiffusionDrivenSIR
    N::Int # population size
    y::Vector{Int} # observations
end

# Ordinary differential equation (ODE) for SIR model with diffusion-driven transmission rate
function sir_ode!(du, u, p, t)
    S, I, R = u[1:3]
    x = u[4]
    N = S + I + R
    γ = p[4]
    β_t = exp(x)
    du[1] = -β_t * I * S / N
    du[2] = β_t * I * S / N - γ * I
    du[3] = γ * I
    du[4] = 0.0 # x is managed by OU process
end

# Compute the PDF of the Ornstein-Uhlenbeck process at time t
function ou_pdf(x, t, ξ1, ξ2, ξ3, x0)
    μ = ξ1 / ξ2
    mean_t = μ + (x0 - μ) * exp(-ξ2 * t)
    variance_t = (ξ3^2) / (2 * ξ2) * (1 - exp(-2 * ξ2 * t))
    return pdf(Normal(mean_t, sqrt(variance_t)), x)
end

# Log density computation
function LogDensityProblems.logdensity(prob::DiffusionDrivenSIR, parameters)
    @unpack N, y = prob
    x₀, s₀, u₀ = parameters[1], parameters[2], parameters[3:end]
    S₀ = s₀ * N
    I₀ = N - S₀ # R₀ = 0 initially
    ξ1, ξ2, ξ3, γ = u₀

    # Prior distributions
    logdensity = sum(logpdf.(Gamma(2, 2), u₀)) +
                 logpdf(Beta(2, 1), s₀) +
                 logpdf(Normal(ξ1 / ξ2, (ξ3^2) / (2 * ξ2)), x₀)

    # Initialize state variables
    S, I, R, W, x = S₀, I₀, 0, 0.0, x₀

    # Loop through time, simulating OU process and SIR ODE
    T = length(y)
    for t in 0.0:T-1
        # Simulate OU process
        ou_process = NoiseProblem(OrnsteinUhlenbeckProcess(ξ1 - ξ2, ξ2, ξ3, t, W), (t, t + 1))
        ou_sol = solve(ou_process, EM(), dt=0.1)
        W = ou_sol.W[end]
        x = ou_sol.u[end]
        logdensity += ou_pdf(x, t, ξ1, ξ2, ξ3, x₀)

        # Simulate SIR ODE
        u = [S, I, R, x]
        p = [ξ1, ξ2, ξ3, γ]
        prob_sir = ODEProblem(sir_ode!, u, (t, t + 1), p)
        sol_sir = solve(prob_sir; dt=1)
        S, I, R = sol_sir.u[end][1:3]

        # Compute log likelihood
        logdensity += logpdf(Poisson(I), y[Int(t)+1])
    end

    return logdensity
end

function LogDensityProblems.dimension(prob::DiffusionDrivenSIR)
    return 6
end

function LogDensityProblems.capabilities(prob::DiffusionDrivenSIR)
    return LogDensityProblems.LogDensityOrder{0}()
end

# Data
y = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4]

prob = DiffusionDrivenSIR(763, y)
params = rand(6)

# Test log density computation
LogDensityProblems.logdensity(prob, params)