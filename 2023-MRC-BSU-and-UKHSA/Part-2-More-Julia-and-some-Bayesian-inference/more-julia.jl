pwd()

using Part2

@which map

pwd()  # current working directory

@which pwd

outputdir(args...) = joinpath("assets", "outputs", "more-julia", args...)

outputdir()

# Create it, if it doesn't exist
mkpath(outputdir())

Threads

Threads.nthreads()

using Base.Threads

nthreads()

Threads.@threads for i in 1:10
    println("Thread $(Threads.threadid()): $i")
end

using Dates
before = Dates.now()

Dates.now() - before

dump(before)

using Random

Random.seed!(1234)

rand()

# Xoshiro is what Julia uses by default
rng = Random.Xoshiro(1234)
rand(rng) # <= same as before

rand(3)

rand(3, 3)

rand(Float32)

randn()

randexp()

# Sample uniformly from a vector
rand([1, 2, 3])

A = [1 2 3; 4 1 6; 7 8 1]

using LinearAlgebra

norm(A), dot(A[:, 1], A[:, 3])

@which norm

using SparseArrays
A_sparse = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])

dropzeros(A_sparse)

# `A` is the dense array from earlier
A * A_sparse

using Statistics

mean(A), std(A)

using Distributed
nprocs()

# Spawn a local worker
addprocs(1)

# Spawn a remote worker (this machine won't work on your computer)
addprocs(["tor@beastly"], tunnel=true, dir="/tmp/")

nprocs()

# Define something on all workers
@everywhere function hostname_and_number(i)
    # Execute shell command on worker to get hostname.
    # NOTE: Using `...` syntax for shell commands.
    # This creates a `Cmd`, which is run once we call `read on it.
    hostname = read(`hostname`, String)
    # Return a tuple of worker ID, hostname and the number.
    return (myid(), i, chomp(hostname))
end

# Run the function on all workers
pmap(hostname_and_number, 1:12)

A = ones(Int, 4, 4)
v = ones(100)
@info "Some variables"  A  s=sum(v)

using Logging
with_logger(NullLogger()) do
    @info "Some variables"  A  s=sum(v)
end

# Some space so you don't cheat.



















# Are you sure?

π

Float64(π)

num_within = 0; num_total = 1_000_000
for i in 1:num_total
    x = 2 .* rand(2) .- 1
    if norm(x) < 1
        num_within += 1
    end
end
# Area of a circle = πr^2 = π * (1/2)^2 = π/4
4 * num_within / num_total

using DataFrames

df = DataFrame(A=1:3, B=5:7, fixed=1)

DataFrame(Dict("A" => 1:3, "B" => 5:7, "fixed" => 1))

df.A

df."A"  # useful when column-names aren't valid Julia symbols

df[:, "A"]

df[:, [:A, :B]]

df[:, Not(:fixed)]

readdir("data")

using CSV
datafile = CSV.File(joinpath("data", "influenza_england_1978_school.csv"));

data = DataFrame(datafile)

using Distributions

dist = Normal()

mean(dist), var(dist)

x = rand(dist)

logpdf(dist, x)

cdf(dist, 0.5)

quantile.(Normal(), [0.05, 0.5, 0.95])

xs = rand(Normal(1, 2), 100)
fit(Normal, xs)

# Filter away abstract types.
nonabstract_dist_subtypes = filter(!isabstracttype, subtypes(Distribution))
# Filter away types which are not found in Distributions.jl.
dist_types_from_distributions = filter(
    Base.Fix1(hasproperty, Distributions) ∘ Symbol,
    nonabstract_dist_subtypes
)

filter(x -> x <: UnivariateDistribution, dist_types_from_distributions)

show(
    IOContext(stdout, :limit => false),
    "text/plain",
    reshape(filter(x -> x <: UnivariateDistribution, dist_types_from_distributions), 10, :)
)

filter(x -> x <: MultivariateDistribution, dist_types_from_distributions)

filter(x -> x <: MatrixDistribution, dist_types_from_distributions)

using Plots

# GR is used by default
Plots.backend()

p1 = plot(1:10, rand(10), size=(450, 200))

p2 = scatter(1:10, rand(10), size=(450, 200))

plot(p1, p2, layout=(1, 2), size=(800, 200))

# Define the Lorenz attractor
Base.@kwdef mutable struct Lorenz
    dt::Float64 = 0.02
    σ::Float64 = 10
    ρ::Float64 = 28
    β::Float64 = 8/3
    x::Float64 = 1
    y::Float64 = 1
    z::Float64 = 1
end

function step!(l::Lorenz)
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z
    l.x += l.dt * dx
    l.y += l.dt * dy
    l.z += l.dt * dz
end

attractor = Lorenz()

# Initialize a 3D plot with 1 empty series
plt = plot3d(
    1,
    xlim = (-30, 30),
    ylim = (-30, 30),
    zlim = (0, 60),
    title = "Lorenz Attractor",
    legend = false,
    marker = 2,
)

# Build an animated gif by pushing new points to the plot, saving every 10th frame
anim = @animate for i=1:1500
    step!(attractor)
    push!(plt, attractor.x, attractor.y, attractor.z)
end every 10
gif(anim, outputdir("lorenz.gif"));

using StatsPlots

plot(Normal())

@df data scatter(:date, :in_bed, label=nothing, ylabel="Number of students in bed")

using DifferentialEquations

const N = 763 # size of population

function SIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    S, I, R = u
    β, γ = p

    du[1] = dS = -β * I * S / N
    du[2] = dI = β * I * S / N - γ * I
    du[3] = dR = γ * I
end

# Include 0 because that's the initial condition before any observations.
tspan = (0.0, 14.0)

# Initial conditions are:
#   S(0) = N - 1; I(0) = 1; R(0) = 0
u0 = [N - 1, 1, 0.0]

# Just to check that everything works, we'll just use some "totally random" values for β and γ:
problem_sir = let β = 2.0, γ = 0.6
    ODEProblem(SIR!, u0, tspan, (β, γ))
end

sol = solve(problem_sir)

plot(
    sol,
    linewidth=2, xaxis="Time in days", label=["Suspectible" "Infected" "Recovered"],
    alpha=0.5, size=(500, 300)
)
scatter!(1:14, data.in_bed, label="Data", color="black")

using Optimization

# Define the loss function.
function loss_sir(problem_orig, p)
    # `remake` just, well, remakes the `problem` with parameters `p` replaced.
    problem = remake(problem_orig, p=p)
    # To ensure we get solutions _exactly_ at the timesteps of interest,
    # i.e. every day we have observations, we use `saveat=1` to tell `solve`
    # to save at every timestep (which is one day).
    sol = solve(problem, saveat=1)
    # Extract the 2nd state, the (I)infected, for the dates with observations.
    sol_for_observed = sol[2, 2:15]
    # Compute the sum-of-squares of the infected vs. data.
    return sum(abs2.(sol_for_observed - data.in_bed))
end

opt_problem = OptimizationProblem(
    OptimizationFunction(
        (p,_) -> loss_sir(problem_sir, p), # function to minimize
        Optimization.AutoForwardDiff()     # use ForwardDiff for automatic differentiation
    ),
    [2.0, 0.5],                            # initial values
    lb = [0, 0],                           # lower bounds on variables
    ub = [Inf, Inf],                       # upper bounds on variables
)

using OptimizationOptimJL
opt = solve(opt_problem, NelderMead())

β, λ = opt
β, λ

# Solve the problem with the obtained parameters.
problem_sir = remake(problem_sir, p=(β, λ))
sol = solve(problem_sir)

# Plot the solution.
plot(sol, linewidth=2, xaxis="Time in days", label=["Susceptible" "Infected" "Recovered"], alpha=0.5)
# And the data.
scatter!(1:14, data.in_bed, label="Data", color="black")

function SEIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    N = 763  # population

    S, E, I, R = u  # have ourselves an additional state!
    β, γ, σ = p     # and an additional parameter!

    # TODO: Implement yah fool!
    du[1] = nothing
    du[2] = nothing
    du[3] = nothing
    du[4] = nothing
end

# Some space so you don't cheat.



















# Are you sure?

function SEIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    N = 763  # population
    S, E, I, R = u  # have ourselves an additional state!
    β, γ, σ = p     # and an additional parameter!

    # Might as well cache these computations.
    βSI = β * S * I / N
    σE = σ * E
    γI = γ * I

    du[1] = -βSI
    du[2] = βSI - σE
    du[3] = σE - γI
    du[4] = γI
end

problem_seir = let u0 = [N - 1, 0, 1, 0], β = 2.0, γ = 0.6, σ = 0.8
    ODEProblem(SEIR!, u0, tspan, (β, γ, σ))
end

sol_seir = solve(problem_seir, saveat=1)

plot(sol_seir, linewidth=2, xaxis="Time in days", label=["Susceptible" "Exposed" "Infected" "Recovered"], alpha=0.5)
scatter!(1:14, data.in_bed, label="Data")

function loss_seir(problem, p)
    problem = remake(problem, p=p)
    sol = solve(problem, saveat=1)
    # NOTE: 3rd state is now the (I)nfectious compartment!!!
    sol_for_observed = sol[3, 2:15]
    return sum(abs2.(sol_for_observed - data.in_bed))
end

opt_problem = OptimizationProblem(
    OptimizationFunction(
        (p,_) -> loss_seir(problem_seir, p), # function to minimize
        Optimization.AutoForwardDiff()       # use ForwardDiff for automatic differentiation
    ),
    [2.0, 0.5, 0.9],                         # initial values
    lb = [0, 0, 0],                          # lower bounds on variables
    ub = [Inf, Inf, Inf],                    # upper bounds on variables
)

opt = solve(opt_problem, NelderMead())

β, γ, σ = opt

sol_seir = solve(remake(problem_seir, p=(β, γ, σ)), saveat=1)
plot(sol_seir, linewidth=2, xaxis="Time in days", label=["Susceptible" "Exposed" "Infected" "Recovered"], alpha=0.5)
scatter!(1:14, data.in_bed, label="Data", color="black")

# Abstract type which we can use to dispatch on.
abstract type AbstractEpidemicProblem end

struct SIRProblem{P} <: AbstractEpidemicProblem
    problem::P
    N::Int
end

function SIRProblem(N::Int; u0 = [N - 1, 1, 0.], tspan = (0, 14), p = [2.0, 0.6])
    return SIRProblem(ODEProblem(SIR!, u0, tspan, p), N)
end

sir = SIRProblem(N);

# General.
parameters(prob::AbstractEpidemicProblem) = prob.problem.p
initial_state(prob::AbstractEpidemicProblem) = prob.problem.u0
population(prob::AbstractEpidemicProblem) = prob.N

# Specializations.
susceptible(::SIRProblem, u::AbstractMatrix) = u[1, :]
infected(::SIRProblem, u::AbstractMatrix) = u[2, :]
recovered(::SIRProblem, u::AbstractMatrix) = u[3, :]

sol = solve(sir.problem, saveat=1)
infected(sir, sol)

struct SEIRProblem <: AbstractEpidemicProblem
    # ...
end

function SEIRProblem end

susceptible
exposed
infected
recovered

# Some space so you don't cheat.



















# Are you sure?

struct SEIRProblem{P} <: AbstractEpidemicProblem
    problem::P
    N::Int
end

function SEIRProblem(N::Int; u0 = [N - 1, 0, 1, 0.], tspan = (0, 14), p = [4.5, 0.45, 0.8])
    return SEIRProblem(ODEProblem(SEIR!, u0, tspan, p), N)
end

susceptible(::SEIRProblem, u::AbstractMatrix) = u[1, :]
exposed(::SEIRProblem, u::AbstractMatrix) = u[2, :]
infected(::SEIRProblem, u::AbstractMatrix) = u[3, :]
recovered(::SEIRProblem, u::AbstractMatrix) = u[4, :]

seir = SEIRProblem(N);
sol = solve(seir.problem, saveat=1)
infected(seir, sol)

function loss(problem_wrapper::AbstractEpidemicProblem, p)
    # NOTE: Extract the `problem` from `problem_wrapper`.
    problem = remake(problem_wrapper.problem, p=p)
    sol = solve(problem, saveat=1)
    # NOTE: Now this is completely general!
    sol_for_observed = infected(problem_wrapper, sol)[2:end]
    return sum(abs2.(sol_for_observed - data.in_bed))
end

loss(SIRProblem(N), [2.0, 0.6])

loss(SEIRProblem(N), [2.0, 0.6, 0.8])

using CUDA

CUDA.has_cuda()

if CUDA.has_cuda()
    CUDA.versioninfo()
end

# Array on CPU
xs = rand(2)

if CUDA.has_cuda()
    # Array on GPU
    xs_cuda = cu(xs)
end

if CUDA.has_cuda()
    2 * xs_cuda
end

if CUDA.has_cuda()
    xs_cuda .+ xs_cuda
end

if CUDA.has_cuda()
    X_cuda = xs_cuda * xs_cuda' + 1f-2 * I
    cholesky(X_cuda)
end

using Turing

# 1. Define the model
@model function simple_demo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end
# 2. Instantiate the model, giving it some data.
model = simple_demo(1.5, 2.0)
# 3. Sample.
chain = sample(model, NUTS(), 1000);

chain

plot(chain)

@model function simple_demo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

model = simple_demo(1.5, 2.0)

model()

@model function simple_demo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))

    # This is just standard Julia, so we can put anything in here.
    return (; s, m, x, y, hello=42)
end

model = simple_demo(1.5, 2.0)
model()

@model function demo_buggy()
    x ~ truncated(Normal(), -10, 0)
    y ~ Normal(0, x)
end
model_buggy = demo_buggy()
model_buggy()

@model function demo_buggy()
    x ~ truncated(Normal(), -10, 0)
    println("x=$x")
    y ~ Normal(0, x)
    println("y=$y")
end
model_buggy = demo_buggy()
model_buggy()

@model function demo_buggy()
    x ~ truncated(Normal(), 0, 10)
    println("x=$x")
    y ~ Normal(0, x)
    println("y=$y")
end
model_buggy = demo_buggy()
model_buggy()

# (1): using the arguments of the function
@model function simple_demo_v1(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end
model_v1 = simple_demo_v1(1.5, 2.0)

# (2): using the `|` operator / `condition`
@model function simple_demo_v2()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end
model_v2 = simple_demo_v2() | (x = 1.5, y = 2.0)

# (3): when `L` is a literal
@model function simple_demo_v3()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
end
model_v3 = simple_demo_v3()

with_logger(NullLogger()) do  # just surpressing the log output for presentation
    chain_v1 = sample(model_v1, NUTS(), 1000; progress=false)
    chain_v2 = sample(model_v2, NUTS(), 1000; progress=false)
    chain_v3 = sample(model_v3, NUTS(), 1000; progress=false)
    plot(chainscat(chain_v1, chain_v2, chain_v3))
end

@model function simple_demo_v1_failure(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))

    x ~ Normal(m, sqrt(s))
    # Used to be: y ~ Normal(m, sqrt(s))
    z = y
    z ~ Normal(m, sqrt(s))
end

model_v1_failure = simple_demo_v1_failure(1.5, 2.0)
model_v1_failure() # `y` is treated as a random variable!!!

model_v1_decondition = simple_demo_v1(1.5, missing)
model_v1_decondition()  # `y` is now a random variable

model_v2_decondition = DynamicPPL.decondition(model_v2, @varname(y))
model_v2_decondition()  # `y` is now a random variable!

logprior(model, (s = 1, m = 1))

loglikelihood(model, (s = 1, m = 1))

logjoint(model, (s = 1, m = 1))

# Condition a variable to be a value
model_with_condition = Turing.condition(model, s=1.0)  # equivalent to `|` operator
model_with_condition()

# Fix a variable to a value
model_with_fixed = Turing.fix(model, s=1.0)
model_with_fixed()

logjoint(model_with_condition, (m=1,)), logjoint(model_with_fixed, (m=1,))

DynamicPPL.observations(model_with_condition)

DynamicPPL.fixed(model_with_fixed)

plot(
    plot(truncated(Normal(2, 1); lower=0), label=nothing, title="β"),
    plot(truncated(Normal(0.4, 0.5); lower=0), label=nothing, title="γ"),
    plot(Exponential(1/5), label=nothing, title="ϕ⁻¹"),
    layout=(3, 1)
)

# `NegativeBinomial` already exists, so let's just make an alternative constructor instead.
function NegativeBinomial2(μ, ϕ)
    p = 1/(1 + μ/ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end

# Let's just make sure we didn't do something stupid.
μ = 2; ϕ = 3;
dist = NegativeBinomial2(μ, ϕ)
# Source: https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
mean(dist) ≈ μ && var(dist) ≈ μ + μ^2 / ϕ

@model function sir_model(
    num_days;                                  # Number of days to model
    tspan = (0.0, float(num_days)),            # Timespan to model
    u0 = [N - 1, 1, 0.0],                      # Initial state
    p0 = [2.0, 0.6],                           # Placeholder parameters
    problem = ODEProblem(SIR!, u0, tspan, p0)  # Create problem once so we can `remake`.
)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)
    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem, p=[β, γ])  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)       # Solve!

    sol_for_observed = sol[2, 2:num_days + 1]  # Timesteps we have observations for.
    in_bed = Vector{Int}(undef, num_days)
    for i = 1:length(sol_for_observed)
        # Add a small constant to `sol_for_observed` to make things more stable.
        in_bed[i] ~ NegativeBinomial2(sol_for_observed[i] + 1e-5, ϕ)
    end

    # Some quantities we might be interested in.
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
end

model = sir_model(length(data.in_bed))

model().infected

# 1. By just calling the `model`, which returns a `NamedTuple` containing the quantities of interest
print(model())

# Sample from prior.
chain_prior = sample(model, Prior(), 10_000);

p = plot(legend=false, size=(600, 300))
plot_trajectories!(p, group(chain_prior, :in_bed); n = 1000)
hline!([N], color="red")

quantities_prior = generated_quantities(model, chain_prior)
print(quantities_prior[1])

# Convert to `Chains`.
chain_quantities_prior = to_chains(quantities_prior);

# Plot.
p = plot(legend=false, size=(600, 300))
plot_trajectories!(p, group(chain_quantities_prior, :infected); n = 1000)
hline!([N], color="red")

p = plot(legend=false, size=(600, 300))
plot_trajectory_quantiles!(p, group(chain_quantities_prior, :infected))
hline!(p, [N], color="red")

DataFrame(quantile(chain_quantities_prior[:, [:R0, :recovery_time], :]))

plot(
    plot(truncated(Normal(2, 1); lower=0), label=nothing, title="β"),
    plot(truncated(Normal(0.4, 0.5); lower=0), label=nothing, title="γ"),
    plot(Exponential(1/5), label=nothing, title="ϕ⁻¹"),
    layout=(3, 1)
)

plot(truncated(Normal(0.4, 0.5); lower=0), label=nothing, title="γ", size=(500, 300))

# ℙ(β > N) = 1 - ℙ(β ≤ N)
1 - cdf(truncated(Normal(2, 1); lower=0), N)

quantile(truncated(Normal(2, 1); lower=0), 0.95)

@model function ModelA()
    x_hidden_from_B ~ Normal()
    x = x_hidden_from_B + 100
    return x
end

@model function ModelB()
    @submodel x = ModelA()
    y ~ Normal(x, 1)

    return (; x, y)
end

# So if we call `B` we only see `x` and `y`
println(ModelB()())

# While if we sample from `B` we get the latent variables
println(rand(ModelB()))

@model ModelA() = (x ~ Normal(); return x + 100)

@model function ModelB()
    # Given it a prefix to use for the variables in `A`.
    @submodel prefix=:inner x_inner = ModelA()
    x ~ Normal(x_inner, 1)

    return (; x_inner, x)
end

print(rand(ModelB()))

@model function prior_original(problem_wrapper::SIRProblem)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)

    return [β, γ]
end

@model function prior_improved(problem_wrapper::SIRProblem)
    # NOTE: Should probably also lower mean for `β` since
    # more probability mass on small `γ` ⟹ `R0 =  β / γ` grows.
    β ~ truncated(Normal(1, 1); lower=0)
    # NOTE: New prior for `γ`.
    γ ~ Beta(2, 5)

    return [β, γ]
end

@model function epidemic_model(
    problem_wrapper::AbstractEpidemicProblem,
    prior  # NOTE: now we just pass the prior as an argument
)
    # NOTE: And use `@submodel` to embed the `prior` in our model.
    @submodel p = prior(problem_wrapper)

    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem_wrapper.problem, p=p)  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)                  # Solve!

    # Extract the `infected`.
    sol_for_observed = infected(problem_wrapper, sol)[2:end]

    # NOTE: `product_distribution` is faster for larger dimensional problems,
    # and it does not require explicit allocation of the vector.
    in_bed ~ product_distribution(NegativeBinomial2.(sol_for_observed .+ 1e-5, ϕ))

    β, γ = p[1:2]
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
end

@model function epidemic_model(
    problem_wrapper::AbstractEpidemicProblem,
    prior  # now we just pass the prior as an argument
)
    # And use `@submodel` to embed the `prior` in our model.
    @submodel p = prior(problem_wrapper)

    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem_wrapper.problem, p=p)  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)                  # Solve!

    # NOTE: Return early if integration failed.
    if !Part2.issuccess(sol)
        Turing.@addlogprob! -Inf  # NOTE: Causes automatic rejection.
        return nothing
    end

    # Extract the `infected`.
    sol_for_observed = infected(problem_wrapper, sol)[2:end]

    # `product_distribution` is faster for larger dimensional problems,
    # and it does not require explicit allocation of the vector.
    in_bed ~ product_distribution(NegativeBinomial2.(sol_for_observed .+ 1e-5, ϕ))

    β, γ = p[1:2]
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
end

sir = SIRProblem(N);
model_original = epidemic_model(sir, prior_original);
model_improved = epidemic_model(sir, prior_improved);

chain_prior_original = sample(model_original, Prior(), 10_000; progress=false);
chain_prior_improved = sample(model_improved, Prior(), 10_000; progress=false);

chain_quantities_original = to_chains(
    generated_quantities(
        model_original,
        chain_prior_original
    );
);

chain_quantities_improved = to_chains(
    generated_quantities(
        model_improved,
        chain_prior_improved
    );
);

p = plot(; legend=false, size=(500, 200))
plot_trajectories!(p, group(chain_quantities_original, :infected); n = 100, trajectory_color="red")
plot_trajectories!(p, group(chain_quantities_improved, :infected); n = 100, trajectory_color="blue")
hline!([N], color="red", linestyle=:dash)

plt1 = plot(legend=false)
plot_trajectory_quantiles!(plt1, group(chain_quantities_original, :infected))
hline!(plt1, [N], color="red", linestyle=:dash)

plt2 = plot(legend=false)
plot_trajectory_quantiles!(plt2, group(chain_quantities_improved, :infected))
hline!(plt2, [N], color="red", linestyle=:dash)

plot(plt1, plt2, layout=(2, 1))

DataFrame(quantile(chain_quantities_improved[:, [:R0, :recovery_time], :]))

DataFrame(quantile(chain_quantities_original[:, [:R0, :recovery_time], :]))

@model function prior_original(problem_wrapper::SEIRProblem)
    # TODO: Implement
end

# Some space so you don't cheat.



















# Are you sure?

@model function prior_original(problem_wrapper::SEIRProblem)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)
    σ ~ truncated(Normal(0.8, 0.5); lower=0)

    return [β, γ, σ]
end

model_seir = epidemic_model(SEIRProblem(N), prior_original)
print(model_seir())

# Condition on the observations.
model = epidemic_model(SIRProblem(N), prior_improved)
model_conditioned = model | (in_bed = data.in_bed,)

chain_mh = sample(model_conditioned, MH(), MCMCThreads(), 10_000, 4; discard_initial=5_000);

plot(chain_mh; size=(800, 500))

# We're using the unconditioned model!
predictions_mh = predict(model, chain_mh)

plot_trajectories!(plot(legend=false, size=(600, 300)), predictions_mh; data=data)

plot_trajectory_quantiles!(plot(legend=false, size=(600, 300)), predictions_mh; data=data)

using SciMLSensitivity

chain = sample(model_conditioned, NUTS(0.8), MCMCThreads(), 1000, 4);

chain

plot(chain; size=(800, 500))

# Predict using the results from NUTS.
predictions = predict(model, chain)

plot_trajectories!(plot(legend=false, size=(600, 300)), predictions; n = 1000, data=data)

plot_trajectory_quantiles!(plot(legend=false, size=(600, 300)), predictions; data=data)

# Sample from the conditioned model so we don't get the `in_bed` variables too
using Random  # Just making sure the numbers of somewhat interesting
rng = MersenneTwister(43);
test_values = rand(rng, NamedTuple, model_conditioned)

model_test = model | test_values

in_best_test = rand(rng, model_test).in_bed;

model_test_conditioned = model | (in_bed = in_best_test,)

# Let's just do a single chain here.
chain_test = sample(model_test_conditioned, NUTS(0.8), 1000);

ps = []
for sym in [:β, :γ, :ϕ⁻¹]
    p = density(chain_test[:, [sym], :])
    vline!([test_values[sym]])
    push!(ps, p)
end
plot(ps..., layout=(3, 1), size=(600, 400))

using LinearAlgebra: I

@model function linear_regression(X)
    num_params = size(X, 1)
    β ~ MvNormal(ones(num_params))
    σ² ~ InverseGamma(2, 3)
    y ~ MvNormal(vec(β' * X), σ² * I)
end

# Generate some dummy data.
X = randn(2, 1_000); lin_reg = linear_regression(X); true_vals = rand(lin_reg)

# Condition.
lin_reg_conditioned = lin_reg | (y = true_vals.y,);

chain_ess_hmc = sample(lin_reg_conditioned, Gibbs(ESS(:β), HMC(1e-3, 16, :σ²)), 1_000);

chain_ess_hmc

@model function mixture(n)
    cluster ~ filldist(Categorical([0.25, 0.75]), n)
    μ ~ MvNormal([-10.0, 10.0], I)
    x ~ product_distribution(Normal.(μ[cluster], 1))
end

model_mixture = mixture(10)
fake_values_mixture = rand(model_mixture)
model_mixture_conditioned = model_mixture | (x = fake_values_mixture.x, )
chain_discrete = sample(
    model_mixture_conditioned, Gibbs(PG(10, :cluster), HMC(1e-3, 16, :μ)), MCMCThreads(), 1_000, 4
)

ps = []
for (i, realizations) in enumerate(eachcol(Array(group(chain_discrete, :cluster))))
    p = density(
        realizations,
        legend=false,
        ticks=false,
        border=:none
    )
    vline!(p, [fake_values_mixture.cluster[i]])
    push!(ps, p)
end
plot(ps..., layout=(length(ps) ÷ 2, 2), size=(600, 40 * length(ps)))

using TuringGLM

register(DataDep(
    "kidiq",
    "Survey of adult American women and their respective children from 2007",
    "https://raw.githubusercontent.com/TuringLang/TuringGLM.jl/bbc9129fc2d1ff7a1026fe2189b6580303d5c9f5/data/kidiq.csv",
))

fname = joinpath(datadep"kidiq", "kidiq.csv")
kidiq = DataFrame(CSV.File(fname))

fm = @formula(kid_score ~ mom_hs * mom_iq)

model = turing_model(fm, kidiq);

chns = sample(model, NUTS(), 1000)

using TuringCallbacks

model = simple_demo(1.5, 2.0);

logdir = mktempdir()

callback = TensorBoardCallback(joinpath(logdir, "logs"); include_hyperparams=true)
chain = sample(model, NUTS(0.8), 1000; callback=callback);

using DataFrames, CSV, DataDeps

register(DataDep(
    "hares-and-lynxes",
    "Numerical data for the number of pelts collected by the Hudson’s Bay Company in the years 1900-1920.",
    "https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/lotka-volterra/hudson-bay-lynx-hare.csv",
))

df = DataFrame(
    CSV.File(
        joinpath(datadep"hares-and-lynxes", "hudson-bay-lynx-hare.csv"),
        skipto=4,
        header=3
    )
)

DataFrame(CSV.File(joinpath("data", "pest_data.csv")))

DataFrame(CSV.File(
    joinpath("data", "time-series.csv")
))

DataFrame(CSV.File(joinpath("data", "influenza_england_1978_school.csv")))

# Without arguments
@which sum

# With arguments
@which sum([1.0])

@macroexpand @model f() = x ~ Normal()

@macroexpand(@model f() = x ~ Normal()) |> Base.remove_linenums!

f(x) = 2x

@code_typed f(1)

@code_llvm f(1)

@code_native f(1)

# Define the Julia function
function mycompare(a, b)::Cint
    println("mycompare($a, $b)")  # NOTE: Let's look at the comparisons made.
    return (a < b) ? -1 : ((a > b) ? +1 : 0)
end

# Get the corresponding C function pointer.
mycompare_c = @cfunction(mycompare, Cint, (Ref{Cdouble}, Ref{Cdouble}))

# Array to sort.
arr = [1.3, -2.7, 4.4, 3.1];

# Call in-place quicksort.
ccall(:qsort, Cvoid, (Ptr{Cdouble}, Csize_t, Csize_t, Ptr{Cvoid}),
      arr, length(arr), sizeof(eltype(arr)), mycompare_c)

# All sorted!
arr

# NOTE: this is NOT `const`, and so it could become some other type
# at any given point without `my_func` knowing about it!
global_variable = 1
my_func_unstable(x) = global_variable * x

using BenchmarkTools
@btime my_func_unstable(2.0);

@code_warntype my_func_unstable(2.0)

const constant_global_variable = 1
my_func_fixed(x) = constant_global_variable * x
@code_warntype my_func_fixed(2.0)

@btime my_func_fixed(2.0);

# HACK: Here we explicitly tell Julia what type `my_func_unstable`
# returns. This is _very_ rarely a good idea because it just hides
# the underlying problem from `@code_warntype`!
my_func_forced(x) = my_func_unstable(x)::typeof(x)
@code_warntype my_func_forced(2.0)

using ProfileView

@profview foreach(_ -> my_func_unstable(2.0), 1_000_000)

ambiguous_function(x, y::Int) = y
ambiguous_function(x::Int, y) = x

# NOTE: Here we have `ambiguous_function(x::Int, y::Int)`
# Which one should we hit?!
ambiguous_function(1, 2)

ambiguous_function(::Int64, ::Int64) = "neato"
ambiguous_function(1, 2)

# NOTE: `@generated` only has access to static information, e.g. types of arguments.
# Here I'm using the special type `Val` to make a number `N` static.
@generated function unrolled_addition(::Val{N}) where {N}
    expr = Expr(:block)
    push!(expr.args, :(x = 0))
    for i = 1:N
        push!(expr.args, :(x += $(3.14 * i)))
    end

    return expr
end

# NOTE: At runtime, it then just returns the result immediately
@code_typed unrolled_addition(Val(10))

@time @eval unrolled_addition(Val(11));

@time @eval unrolled_addition(Val(10_001));

@btime unrolled_addition(Val(10_001));

function not_unrolled_addition(N)
    x = 0
    for i = 1:N
        x += 3.14 * i
    end

    return x
end

@btime not_unrolled_addition(10_001);

@generated function unrolled_addition_old(::Val{N}) where {N}
    expr = Expr(:block)
    push!(expr.args, :(x = 0))
    for i = 1:N
        push!(expr.args, :(x += $i))  # NOTE: No 3.14!
    end
    return expr
end
function not_unrolled_addition_old(N)
    x = 0
    for i = 1:N
        x += i  # NOTE: No 3.14!
    end
    return x
end

@btime unrolled_addition_old(Val(10_001));
@btime not_unrolled_addition_old(10_001);

# NOTE: The one LLVM failed to unroll
@code_llvm not_unrolled_addition(10_001)

# NOTE: The one LLVM seems to have unrolled.
@code_llvm not_unrolled_addition_old(10_001)
