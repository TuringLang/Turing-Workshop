using JuliaBUGS

using Distributions
using DifferentialEquations

using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, AdvancedHMC
using MCMCChains

# Automatic differentiation
using ReverseDiff
using ForwardDiff

# Demo 1: classic examples: rats
rats_model = JuliaBUGS.BUGSExamples.rats.model_def;
data = JuliaBUGS.BUGSExamples.rats.data;
inits = JuliaBUGS.BUGSExamples.rats.inits[1];
model = compile(rats_model, data, inits);

# Inference
ad_model = ADgradient(:ReverseDiff, model; compile = Val(true))

n_samples = 3000;
n_adapts = 1000;

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

# Plot with Makie
# Makie is a large package, so we didn't include it in the Project.toml
# you can install it with `] add Makie, GraphMakie`
# using GLMakie, GraphMakie
# graphplot(model)

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


data=(
        x=[8.0, 15.0, 22.0, 29.0, 36.0],
        xbar=22,
        N=2,
        T=5,
        Y=[
            151 199 246 283 320
            145 199 249 293 354
        ],
    )

    inits = (
        alpha=ones(Integer, 2) .* 250,
        beta=ones(Integer, 2) .* 6,
        var"alpha.c"=150,
        var"beta.c"=10,
        var"tau.c"=1,
        var"alpha.tau"=1,
        var"beta.tau"=1,
    )



using JuliaBUGS

using Distributions
using DifferentialEquations

using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, AdvancedHMC
using MCMCChains

# Automatic differentiation
using ReverseDiff
using ForwardDiff

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

# Plot with Makie
# Makie is a large package, so we didn't include it in the Project.toml
# you can install it with `] add Makie, GraphMakie`
using GLMakie, GraphMakie
using NetworkLayout
using GraphPlot
using TikzGraphs

f, ax, p = graphplot(model; layout =Spring(; dim=2))
gplot(model)
TikzGraphs.plot(model)

using AdvancedMH

# use `ForwardDiff` this time
ad_model = ADgradient(:ForwardDiff, model)

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    AbstractMCMC.MCMCThreads(),
    n_samples,
    4;
    chain_type = Chains,
    n_adapts = n_adapts,
    init_params = [initial_θ for _ in 1:4],
    discard_initial = n_adapts,
)

n_samples = 3000;
n_adapts = 1000;

initial_θ = rand(LogDensityProblems.dimension(model))

samples_and_stats = AbstractMCMC.sample(
    model,
    AdvancedMH.RWMH(MvNormal(zeros(3), I)),
    AbstractMCMC.MCMCThreads(),
    n_samples,
    4
    ;
    chain_type = Chains,
    n_adapts = n_adapts,
    init_params = [initial_θ for _ in 1:4],
    discard_initial = n_adapts,
)

using LinearAlgebra

spl = RWMH(MvNormal(zeros(3), I))
sample(model, spl, 100000)

AdvancedMH.MvNormal(zeros(3), I) isa AbstractMCMC.AbstractSampler
NUTS(0.65) isa AbstractMCMC.AbstractSampler

D = 3; initial_θ = rand(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ad_model, ForwardDiff)

# Define a leapfrog solver, with the initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true)


println(Threads.nthreads())
nchains = 4

using BenchmarkTools
using JuliaBUGS
x = 1; y = 2
@btime eval(:(x + JuliaBUGS.exp(y)))

expr = :(x + JuliaBUGS.exp(y))
expr.args[1] = +
expr.args[2]
expr.args[3].args[1] = JuliaBUGS.exp

@btime eval(expr)

@btime JuliaBUGS._eval(:(x + JuliaBUGS.exp(y)), Dict(:x => x, :y => y))
@btime x + JuliaBUGS.exp(y)