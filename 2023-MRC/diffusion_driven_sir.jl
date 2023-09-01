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
function ou_logdensity(x, t, ξ1, ξ2, ξ3, x0)
    μ = ξ1 / ξ2
    mean_t = μ + (x0 - μ) * exp(-ξ2 * t)
    variance_t = (ξ3^2) / (2 * ξ2) * (1 - exp(-2 * ξ2 * t))
    return logpdf(Normal(mean_t, sqrt(variance_t)), x)
end

# Log density computation
function LogDensityProblems.logdensity(prob::DiffusionDrivenSIR, parameters)
    @unpack N, y = prob
    x₀, s₀, u₀, ws = parameters[1], parameters[2], parameters[3:6], parameters[7:end]

    ws = reshape(ws, length(y), 10)
    S₀ = s₀ * N
    I₀ = N - S₀ # R₀ = 0 initially
    ξ1, ξ2, ξ3, γ = u₀

    # Prior distributions
    logdensity = sum(logpdf.(Gamma(2, 2), u₀)) +
                 logpdf(Beta(2, 1), s₀) +
                 logpdf(Normal(ξ1 / ξ2, (ξ3^2) / (2 * ξ2)), x₀)

    # Initialize state variables
    S, I, R, x = S₀, I₀, 0, x₀

    # Loop through time, simulating OU process and SIR ODE
    T = length(y)
    for t in 0.0:T-1
        # Simulate OU process
        simulated_x = simulate_OU(ξ1, ξ2, ξ3, x, 9, ws[Int(t)+1, :])
        x = simulated_x[end]
        # logdensity += ou_logdensity(x, t, ξ1, ξ2, ξ3, x₀)
        logdensity += cumsum(logpdf.(Normal(), ws[Int(t)+1, :]))

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



function simulate_OU(ξ1, ξ2, ξ3, X0, l, ws_t)
    dt = 1 / (l + 1)
    N = l + 1
    X = Vector{typeof(X0)}(undef, N+1)
    X[1] = X0

    for i in 2:N+1
        dW = ws_t[i-1] * sqrt(dt)
        drift = (ξ1 - ξ2 * X[i-1]) * dt
        diffusion = ξ3 * dW
        X[i] = X[i-1] + drift + diffusion
    end

    return X
end


function LogDensityProblems.dimension(prob::DiffusionDrivenSIR)
    return 6 + length(prob.y) * 10
end

function LogDensityProblems.capabilities(prob::DiffusionDrivenSIR)
    return LogDensityProblems.LogDensityOrder{0}()
end

# Data
y = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4]

prob = DiffusionDrivenSIR(763, y)
params = rand(LogDensityProblems.dimension(prob))

# Test log density computation
LogDensityProblems.logdensity(prob, params)

using Turing
using ReverseDiff
using Zygote
Turing.setadbackend(:forwarddiff)

@model function DiffusionDrivenSIRModel(y, N, ::Type{T} = Float64) where T
    # Hyperparameters for ξ1, ξ2, ξ3, γ
    ξ1 ~ Gamma(2, 2)
    ξ2 ~ Gamma(2, 2)
    ξ3 ~ Gamma(2, 2)
    γ ~ Gamma(2, 2)
    
    # Prior for s₀
    s₀ ~ Beta(2, 1)

    # Prior for x₀ based on the stationary distribution of the OU diffusion
    x₀ ~ Normal(ξ1 / ξ2, (ξ3^2) / (2 * ξ2))

    # Noise
    ws ~ filldist(Normal(), length(y) * 10)

    ws_matrix = reshape(ws, length(y), 10)
    S₀ = s₀ * N
    I₀ = N - S₀

    # Initialize state variables
    S, I, R, x = S₀, I₀, 0, x₀

    # Loop through time, simulating OU process and SIR ODE
    num_times = length(y)
    for t in 0.0:num_times-1
        # Simulate OU process
        simulated_x = simulate_OU(ξ1, ξ2, ξ3, x, 9, ws_matrix[Int(t)+1, :])
        x = simulated_x[end]

        # Simulate SIR ODE
        u = [S, I, R, x]
        p = [ξ1, ξ2, ξ3, γ]
        prob_sir = ODEProblem(sir_ode!, u, (t, t + 1), p)
        sol_sir = solve(prob_sir; dt=1)
        S, I, R = sol_sir.u[end][1:3]

        # Observe the data
        y[Int(t)+1] ~ Poisson(I)
    end
end

model = DiffusionDrivenSIRModel(y, 763)

rand(model)

samples = sample(model, HMC(0.1, 25), 3000; nadapts = 1000)