# Now let's do a more complex model
# Hierarchical Linear Regression
using Turing
using LinearAlgebra: I
using Statistics: mean, std

@model function varying_intercept(
    X,
    idx,
    y;
    n_gr = length(unique(idx)),
    predictors = size(X, 2),
)
    # priors
    α ~ Normal(0, 2.5)                      # population-level intercept
    β ~ filldist(Normal(0, 2), predictors)  # population-level coefficients
    σ ~ Exponential(1)                      # residual SD

    # prior for variance of random intercepts
    # usually requires thoughtful specification
    τ ~ truncated(Cauchy(0, 2); lower = 0)    # group-level SDs intercepts
    αⱼ ~ filldist(Normal(0, τ), n_gr)         # group-level intercepts

    # likelihood
    ŷ = α .+ X * β .+ αⱼ[idx]
    y ~ MvNormal(ŷ, σ^2 * I)
end;

# There's a famous dataset called cheese which is data from cheese ratings.
# A group of 10 rural and 10 urban raters rated
# 4 types of different cheeses (A, B, C and D) in two samples.
using DataFrames
using CSV

file = joinpath(@__DIR__, "data", "cheese.csv") # compatible with different filesystems
cheese = CSV.read(file, DataFrame)
describe(cheese)

# A nice shameless plug for data science in Julia is the open book:
# https://juliadatascience.io/
# In order to prepare the data for Turing, I will convert the Strings in variables cheese
# and background to Ints. Regarding cheese, I will create 4 dummy variables one for each cheese type;
# and background will be converted to integer data taking two values:
# one for each background type.
#
# My intent is to model background as a group for intercepts.
for c in unique(cheese[:, :cheese])
    cheese[:, "cheese_$c"] = ifelse.(cheese[:, :cheese] .== c, 1, 0)
end

cheese[:, :background_int] = map(cheese[:, :background]) do b
    if b == "rural"
        1
    elseif b == "urban"
        2
    else
        missing
    end
end

first(cheese, 5)

# Now let's us instantiate our model with the data:
X = Matrix(select(cheese, Between(:cheese_A, :cheese_D)))
y = cheese[:, :y]
y_standard = (y .- mean(y)) ./ std(y)
idx = cheese[:, :background_int]

model_cheese = varying_intercept(X, idx, y_standard)

# and let's sample from it
chain_cheese = sample(model_cheese, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_cheese)
