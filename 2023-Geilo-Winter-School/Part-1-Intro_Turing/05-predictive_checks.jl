# Let's recall our model from the intro to Turing session
using Distributions
using Random
using Turing

Random.seed!(123)
dice = Categorical([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
my_data = rand(dice, 1_000)

@model function dice_throw(N)
    p ~ Dirichlet(6, 1) # a uniform prior
    y = Vector{Int}(undef, N)
    y ~ filldist(Categorical(p), N)
end
# instantating a a model
my_model = dice_throw(1_000)
# "condition" the model with data
my_model_cond = my_model | (y = my_data,)
my_chains_nuts = sample(my_model_cond, NUTS(), MCMCThreads(), 1_000, 2)

# Prior Check
# just use the Prior() sampler:
prior_chain = sample(my_model_cond, Prior(), 2_000)

# This is a regular MCMCChains object,
# you can plot it:
using MCMCChains
using StatsPlots
plot(prior_chain)

# Prior Predictive Check
# we just use the "unconditioned model":
# Now we call predict() on the predictive model and the prior samples:
prior_check = predict(my_model, prior_chain)
summarystats(prior_check[:, 1:5, :]) # just the first 5 prior samples

# Posterior Predictive Checks
# the same ideia but not we use the normal chain and not the prior chain:
posterior_check = predict(my_model, my_chains_nuts)
summarystats(posterior_check[:, 1:5, :]) # just the first 5 posterior samples
# You can also plot
plot(posterior_check[:, 1:3, 1]) # just the first 3 posterior samples of a single chain