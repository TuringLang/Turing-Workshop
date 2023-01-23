# Let's recall our model from the intro to Turing session
using Distributions
using Random
using Turing

Random.seed!(123)
dice = Categorical([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
my_data = rand(dice, 1_000)
@model function dice_throw(y)
    p ~ Dirichlet(6, 1) # a uniform prior
    for i in eachindex(y)
        y[i] ~ Categorical(p)
    end
end
my_model = dice_throw(my_data)
my_chains_nuts = sample(my_model, NUTS(), MCMCThreads(), 1_000, 2)

# You can get a summary of the internals with summary statistics mean and quantile
# using kwarg sections=:internals
mean(my_chains_nuts; sections = :internals)
quantile(my_chains_nuts; sections = :internals)

# Now for diagnostics.
# All of the diagnostics are available in MCMCDiagnostics.jl
# Check them in: https://turinglang.github.io/MCMCDiagnosticTools.jl/stable/
# MCMCDiagnostics is automatically loaded with Turing
ess_rhat(my_chains_nuts)
