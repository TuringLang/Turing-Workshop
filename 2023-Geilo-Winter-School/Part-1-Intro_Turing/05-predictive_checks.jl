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

# Prior Predictive Check
# just use the Prior() sampler:
prior_chain = sample(my_model, Prior(), 2_000)

# This is a regular MCMCChains object,
# you can plot it:
using MCMCChains
using StatsPlots
plot(prior_chain)

# we instantiate a "predictive model",
# i.e. a Turing model but with the observations set to missing:
missing_data = similar(y, Missing) # vector of `missing`
model_missing = dice_throw(missing_data) # instantiate the predictive model

# Now we call predict() on the predictive model and the prior samples:
prior_check = predict(model_missing, prior_chain)
summarystats(prior_check[:, 1:5, :]) # just the first 5 prior samples

# Posterior Predictive Checks
# the sssame ideia but not we use the normal chain and not the prior chain:
posterior_check = predict(model_missing, my_chains_nuts)
summarystats(posterior_check[:, 1:5, :]) # just the first 5 posterior samples
