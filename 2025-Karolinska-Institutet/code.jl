## Data

using DataFrames, RDatasets
using Statistics: mean, std

df_lung = dataset("survival", "cancer")
DataFrames.describe(df_lung)

vars_to_check = [:Time, :Status, :Age, :Sex]
df = dropmissing(df_lung, vars_to_check)

standardise(x) = (x .- mean(x)) ./ std(x)
age_std = standardise(df.Age)

female = df.Sex .== 2

# Create design matrix X with intercept, standardized Age, and Female indicator
X = hcat(ones(nrow(df)), age_std, female)

y = Float64.(df.Time)
censoring_time = y

# Create event indicator (true if event/death occurred, false if censored)
event = df.Status .== 2;

## A first model

using Turing

@model function aft_weibull_model_initial(X, event, censoring_time)
    n, p = size(X)
    y = Vector{Real}(undef, n)

    α ~ LogNormal(0, 2) # Weibull shape parameter

    # Weibull scale parameter for every individual (θᵢ = exp(xᵢᵀβ))
    β ~ MvNormal(zeros(p), 10.0I)
    θ = exp.(X * β)

    for i = 1:n
        if event[i] # death
            y[i] ~ Weibull(α, θ[i])
        else
            y[i] ~ censored(Weibull(α, θ[i]); upper = censoring_time[i])
        end
    end

    return (; α, β, y)
end

model_initial = aft_weibull_model_initial(X, event, censoring_time)

# Simulate from the prior, return the expressions specified by `return` expression
sampled_return_values = model_initial()

# Simulate from the prior, return the samples of the model parameters
sample_of_model_parameters = rand(model_initial)

## Prior predictive check

# Sample from the prior predictive distribution
prior_samples = sample(model_initial, Prior(), 100)

# Get returned values with the sampled parameters
returned_values = returned(model_initial, prior_samples)

plot_prior_pred_check(model_initial, prior_samples, y, event)

## Adjusted prior
using LinearAlgebra

@model function aft_weibull_model_adjusted(X, event, censoring_time)
    n, p = size(X)
    y = Vector{Real}(undef, n)

    α ~ LogNormal(log(0.8), 0.5)

    μβ = [log(500.0); zeros(p - 1)]
    σβ = Diagonal(vcat([1.0^2], fill(5.0^2, p - 1)))
    β ~ MvNormal(μβ, σβ)

    θ = exp.(X * β)

    for i = 1:n
        if event[i]
            y[i] ~ Weibull(α, θ[i])
        else
            y[i] ~ censored(Weibull(α, θ[i]); upper = censoring_time[i])
        end
    end

    return (; α, β, y)
end

model_adjusted = aft_weibull_model_adjusted(X, event, censoring_time)
prior_samples_adjusted = sample(model_adjusted, Prior(), 100)
returned_values_adjusted = returned(model_adjusted, prior_samples_adjusted)
plot_prior_pred_check(model_adjusted, prior_samples_adjusted, y, event)

## Debugging model

@model function aft_weibull_model_adjusted_debug(X, event, censoring_time)
    n, p = size(X)
    y = Vector{Real}(undef, n)

    α ~ LogNormal(log(0.8), 0.5)

    μβ = [log(500.0); zeros(p - 1)]
    σβ = Diagonal(vcat([1.0^2], fill(5.0^2, p - 1)))
    β ~ MvNormal(μβ, σβ)

    θ = exp.(X * β)

    println("α: ", α)
    println("β: ", β)
    println("θ: ", θ)

    for i = 1:n
        if event[i]
            y[i] ~ Weibull(α, θ[i])
        else
            y[i] ~ censored(Weibull(α, θ[i]); upper = censoring_time[i])
        end
    end

    return (; α, β, y)
end

model_debug = aft_weibull_model_adjusted_debug(X, event, censoring_time)
model_debug()

DynamicPPL.DebugUtils.check_model_and_trace(model_debug)
DynamicPPL.DebugUtils.model_warntype(model_debug)

## `condition` operator

conditioned_model = model_adjusted | (; y = y)

rand(conditioned_model)

## Posterior sampling

using ReverseDiff # Mooncake is also very good!

# Draw 500 samples from the posterior, 6 chains, using the NUTS sampler with reverse-mode AD
posterior_chains = sample(
    conditioned_model,
    NUTS(; adtype = AutoReverseDiff(; compile = true)),
    MCMCThreads(),
    500,
    6;
    n_adapt = 500,
)

## Trace and rank plots

using StatsPlots, Plots

# MCMCChains integrates with StatsPlots
plot(posterior_chains)

using ArviZ, ArviZPythonPlots

use_style("arviz-darkgrid")

idata = ArviZ.from_mcmcchains(posterior_chains)
# plot_trace(idata)
plot_rank(idata)
gcf()

## fix operator

fixed_model = fix(model_adjusted, (; α = 0.5))

rand(fixed_model) # the difference is that the log density contributions from the fixed parameters are 0

## Posterior predictive check

plot_posterior_pred_check(model_adjusted, posterior_chains, y, event)

## LOO and LOO-PIT

# Extract parameter samples from the chain
parameter_samples = MCMCChains.get_sections(posterior_chains, :parameters)

# Compute pointwise log likelihoods (returns an `OrderedDict`)
log_likelihood_dict = DynamicPPL.pointwise_loglikelihoods(
    conditioned_model, # used the conditioned model for pointwise log likelihoods
    parameter_samples,
)

# Compute posterior predictive samples
posterior_predictive_chain = DynamicPPL.predict(
    model_adjusted, # Use the unconditioned model for prediction
    parameter_samples,
)

# Reformat the log likelihoods into an array suitable for ArviZ
# Dimensions: (draws, chains, observations)
n_draws, n_params, n_chains = size(posterior_chains)
n_obs = length(y)
log_lik_values = Array{Float64}(undef, n_draws, n_chains, n_obs)
post_pred_values = Array{Float64}(undef, n_draws, n_chains, n_obs)

for i = 1:n_obs
    vn_str = "y[$i]"
    vn_sym = Symbol(vn_str)

    log_lik_values[:, :, i] .= log_likelihood_dict[vn_str]
    post_pred_values[:, :, i] .=
        dropdims(posterior_predictive_chain[[vn_sym],].value, dims = 2)
end

log_likelihood_data = (; y = log_lik_values)
posterior_predictive_data = (; y = post_pred_values)

idata_loo = ArviZ.from_mcmcchains(
    posterior_chains;
    log_likelihood = log_likelihood_data,
    posterior_predictive = posterior_predictive_data,
    observed_data = (; y = y),
    coords = (obs_id = 1:n_obs, coef_dim = 1:size(X, 2)),
    dims = (y = [:obs_id], β = [:coef_dim]),
)

loo_result = ArviZ.loo(idata_loo)

log_weights = loo_result.psis_result.log_weights
loo_pit_values = ArviZ.loo_pit(idata_loo, log_weights)

plot_loo_pit(idata_loo; y = "y", ecdf = true, color = "maroon")
gcf()

## Model with individual frailty (vectorized submodel approach)

@model function vectorized_gamma_frailties_submodel(n::Int, k::Real)
    frailties ~ Turing.arraydist([Gamma(k, k) for _ = 1:n])
end

@model function aft_weibull_frailty_age_sub(X, event, censoring_time)
    n, p = size(X)

    α ~ LogNormal(0, 1.5)
    β ~ MvNormal(zeros(p), 10I)
    k ~ Gamma(2, 2)

    v = Vector{Real}(undef, n)
    v ~ to_submodel(vectorized_gamma_frailties_submodel(n, k))

    θ_base = exp.(X * β)

    y = Vector{Real}(undef, n)
    for i = 1:n
        d = Weibull(α, θ_base[i] / (v[i]^(1 / α)))

        if event[i]
            y[i] ~ d
        else
            y[i] ~ censored(d; upper = censoring_time[i])
        end
    end

    return (; α, β, v, k, y)
end

model_frailty = aft_weibull_frailty_age_sub(X, event, censoring_time)
conditioned_model_frailty = model_frailty | (; y = y)

rand(conditioned_model_frailty)

posterior_chains_frailty = sample(
    conditioned_model_frailty,
    NUTS(; adtype = AutoReverseDiff(; compile = true)),
    MCMCThreads(),
    1000,
    6;
    n_adapt = 500,
)

plot_posterior_pred_check_frailty(
    model_frailty,
    posterior_chains_frailty,
    y,
    event,
)
