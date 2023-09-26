using MyResearch

data = MyResearch.load_data()

using Turing, DifferentialEquations, StatsPlots

N = 763

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

# plot(
#     plot(truncated(Normal(2, 1); lower=0)),
#     plot(truncated(Normal(0.4, 0.5); lower=0)),
#     layout=(2, 1)
# )


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

    # Check that `solve` was a success.
    # If it wasn't, we just reject this sample.
    if sol.retcode != ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end

    sol_for_observed = sol[2, 2:num_days + 1]  # Timesteps we have observations for.
    in_bed = Vector{Int}(undef, num_days)
    for i = 1:length(sol_for_observed)
        # Add a small constant to `sol_for_observed` to make things more stable.
        in_bed[i] ~ NegativeBinomial2(sol_for_observed[i] + 1e-5, ϕ)
    end
end

in_bed_subset = data.in_bed[1:8]
model = sir_model(7) | (in_bed = in_bed_subset,)
ret = model();
ret.sol.retcode


# Sample.
chain = sample(model, NUTS(), 1000)
mean(chain)


### Create a model where we can start
# sampling from previous last state.
variables_samples = sample(chain, 10)
gen_qts = generated_quantities(model, variables_samples);
length(gen_qts)

# A single generated quantity to check that it works.
x = gen_qts;
keys(x)
sol = x.sol
u0 = sol[end]
u0

in_bed_next = data.in_bed[9:end]
model_next = sir_model(length(in_bed_next); u0=u0) | (in_bed=in_bed_next,)
chain_next = sample(model_next, NUTS(), 1000)


# All of them.
using ProgressMeter

chains = []
@showprogress for genval in gen_qts
    u0 = genval.sol[end]
    model_next = sir_model(length(in_bed_next); u0=u0) | (in_bed=in_bed_next,)
    chain_next = sample(model_next, NUTS(), 1000; progress=false)
    push!(chains, chain_next)
end
length(chains)
chains_combined = chainscat(chains...)

### Conditioning some variables
model_unconditioned = sir_model(14);
rand(DynamicPPL.OrderedDict, model_unconditioned)

model_conditioned = model_unconditioned | (@varname(in_bed[1]) => 3, @varname(in_bed[2]) => 5);
rand(DynamicPPL.OrderedDict, model_conditioned)

ret = model();
ret.sol.retcode


@model function sir_model_fast(
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

    # Check that `solve` was a success.
    # If it wasn't, we just reject this sample.
    if sol.retcode != ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end

    sol_for_observed = sol[2, 2:num_days + 1]  # Timesteps we have observations for.
    in_bed ~ product_distribution(NegativeBinomial2.(sol_for_observed .+ 1e-5, ϕ))
end

length(in_bed_subset)
model_fast = sir_model_fast(7) | (in_bed = in_bed_subset[1:end - 1],)
chain_fast = sample(model_fast, NUTS(), 1000);

model = sir_model(100; tspan=(0, 100))
chain_preds = predict(model, chain_fast)

size(chain_preds)

f(x) = x^2
@time f(1)

@time f.(1:100)

@time model();

using BenchmarkTools
@benchmark $model()

model()

using TuringBenchmarking
results = benchmark_model(model_fast; adbackend=[:ForwardDiff,])


# TuringCallbacks.jl: viewing trace in TensorBoard
using TuringCallbacks
callback = TensorBoardCallback("logs")
chain_fast = sample(model_fast, NUTS(), 1000; callback=callback);
