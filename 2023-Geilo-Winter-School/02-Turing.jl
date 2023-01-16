# Let's generate some fake data from  a biased six-sided dice
using Distributions
using Random

Random.seed!(123)

dice = Categorical([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])

mean(dice)
mode(dice)

my_data = rand(dice, 1_000)

# Now let's see how we define a model using Turing
using Turing

# we use the @model macro to annotate a function
@model function dice_throw(y)
    # Inside we have probabilistic statements using ~
    # or deterministic statements using =

    # let's define a random variable p that will act
    # as a parameter and a distribution as a prior
    p ~ Dirichlet(6, 1) # a uniform prior

    # Since we are supplying y, if we do a ~
    # assignment it will be threated as a likelihood
    for i in eachindex(y)
        y[i] ~ Categorical(p)
    end
end

# With the model defined, we can instantiate a model with our fake data as y:
my_model = dice_throw(my_data)

# Now that the model is instantiated, we can use sample() on it.
# sample() takes:
# 1. model
# 2. sampler
# 3. number of samples
# optional. parallel algorithm (multi-threaded or multi-process)
# optional. number of chains
my_chain_mh = sample(my_model, MH(), 1_000)
my_chain_nuts = sample(my_model, NUTS(), 1_000)

# Multiple dispatch sample with the optional parallel algorithm
# and number of chains
my_chains_mh = sample(my_model, MH(), MCMCThreads(), 1_000, 4)
my_chains_nuts = sample(my_model, NUTS(), MCMCThreads(), 1_000, 4)

# We can inspect chains with describe().
# It returns two DataFrames:
# 1. summarystats - summary statistics and diagnostics
# 2. quantile - quantiles
summaries, quantiles = describe(my_chains_nuts)

# We can also get those separately by calling either
# summarystats() or quantile()
summarystats(my_chains_nuts)
quantile(my_chains_nuts)

# Visualizations
# We can use MCMCCHains.jl to plot chains
using MCMCChains
using StatsPlots

# there are several visualizations
# check https://turinglang.github.io/MCMCChains.jl/stable/statsplots/
plot(my_chain_nuts)
