using JuliaBUGS

using Distributions
using DifferentialEquations

using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, AdvancedHMC
using MCMCChains

# Automatic differentiation
using ReverseDiff
using ForwardDiff

using GraphPlot

# Demo 1: classic examples: rats
rats_model = JuliaBUGS.BUGSExamples.rats.model_def;
data = JuliaBUGS.BUGSExamples.rats.data;
inits = JuliaBUGS.BUGSExamples.rats.inits[1];
model = compile(rats_model, data, inits);

# use a smaller model to demonstrate plotting
demo_data = begin 
   data_dict = Dict(pairs(data))
   data_dict[:Y] = data_dict[:Y][1:3, :]
   NamedTuple(data_dict)
end
demo_inits = begin
    inits_dict = Dict(pairs(inits))
    inits_dict[:alpha] = ones(Integer, 3) .* 250
    inits_dict[:beta] = ones(Integer, 3) .* 6
    NamedTuple(demo_inits)
end
demo_model = compile(rats_model, demo_data, demo_inits)
gplot(demo_model)

# Inference
ad_model = ADgradient(:ReverseDiff, model; compile = Val(true))

n_samples = 3000; n_adapts = 1000

initial_θ = rand(LogDensityProblems.dimension(model))

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    n_samples;
    chain_type = Chains,
    n_adapts = n_adapts,
    init_params = initial_θ,
    discard_initial = n_adapts,
)

# Demo 2: SIR model

# english_boarding_schools from `outbreaks`
I_data = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4]

# same as Tor's
function SIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t,    # current time
)
    N = 763  # population
    S, I, R = u
    β, γ = p

    du[1] = dS = -β * I * S / N
    du[2] = dI = β * I * S / N - γ * I
    du[3] = dR = γ * I
end

# registering a distribution
JuliaBUGS.@register_primitive function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end

# another way to register a primitive
function solve_ode(u0, p)
    tspan = (0.0, 14.0)
    prob = DifferentialEquations.ODEProblem(SIR!, u0, tspan, p)
    sol = solve(prob; saveat = 1.0)
    return sol[2, 2:15]
end

JuliaBUGS.@register_primitive solve_ode

sir_bugs_model = @bugs begin
    β ~ truncated(Normal(2, 1), 0, nothing)
    γ ~ truncated(Normal(0.4, 0.5), 0, nothing)
    ϕ⁻¹ ~ Exponential(1 / 5)
    ϕ = inv(ϕ⁻¹)

    p[1] = β
    p[2] = γ

    predicted[1:14] = solve_ode(u0[:], p[:])
    for i = 1:14
        I_data[i] ~ NegativeBinomial2(predicted[i] + 1e-5, ϕ)
    end

    # generated quantities
    R0 = β / γ
    recovery_time = 1 / γ
    infected[1:14] = predicted[:]
end
data = Dict(:I_data => I_data, :u0 => [762.0, 1.0, 0.0])
inits = (; β = 2, γ = 0.5, ϕ⁻¹ = 0.2)
model = compile(sir_bugs_model, data, inits);

# use `ForwardDiff` this time
ad_model = ADgradient(:ForwardDiff, model)

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    n_samples;
    chain_type = Chains,
    n_adapts = n_adapts,
    init_params = [1.6, 7, 1.2],
    discard_initial = n_adapts,
)
