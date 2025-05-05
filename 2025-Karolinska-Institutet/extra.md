## heavier tails (Log‑Logistic baseline)

### Log‑Logistic AFT model

```julia
using Turing, Distributions, LinearAlgebra, Random


# Simple positive‑support Log‑Logistic distribution
struct LogLogistic{T<:Real} <: ContinuousUnivariateDistribution
    α::T   # shape (α > 0)
    θ::T   # scale (θ > 0)
end
Distributions.support(d::LogLogistic) = RealInterval(0.0, Inf)
Base.rand(rng::AbstractRNG, d::LogLogistic) =
    d.θ * (rand(rng, Logistic(0,1)) |> exp)^(1/d.α)
Distributions.logpdf(d::LogLogistic, t::Real) =
    log(d.α) - log(d.θ) + (d.α - 1) * log(t / d.θ) - 2log1p((t / d.θ) ^ d.α)

# CDF is required for `censored` observations in Turing/DynamicPPL
function Distributions.cdf(d::LogLogistic, t::Real)
    t < 0 && return zero(t)                     # support(d) = (0, ∞)
    1 / (1 + (d.θ / t)^d.α)                     # F(t) = 1 / (1 + (θ/t)^α)
end


@model function aft_loglogistic_model(X, event, y_obs)
    # Dimensions
    n, p = size(X)

    # Storage
    y = Vector{Float64}(undef, n)

    # Priors
    α  ~ LogNormal(0, 1)                    # tail‐shape
    β  ~ MvNormal(zeros(p), 10I)            # regression coefficients

    # Individual scale parameters on the original time scale
    θ = exp.(X * β)                         # vector length n

    # Likelihood with right‑censoring
    for i in 1:n
        d = LogLogistic(α, θ[i])
        if event[i]
            y[i] ~ d                        # observed death
        else
            y[i] ~ censored(d; upper = y_obs[i])  # right‑censored
        end
    end

    return (; α, β, y)
end
```

### Posterior sampling

```julia
# Build and condition the model
model_ll             = aft_loglogistic_model(X, event, y)
conditioned_model_ll = model_ll | (; y = y)

# Draw posterior samples (6 chains × 1000 draws, 500 warm‑up)
posterior_chains_ll = sample(
    conditioned_model_ll,
    # NUTS(; adtype = AutoReverseDiff(; compile = true)),
    NUTS(; adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(),
    1000, 6; n_adapt = 500
)
```

### Posterior predictive check

```julia
function plot_posterior_pred_check_ll(
    model::DynamicPPL.Model,
    posterior_samples::MCMCChains.Chains,
    y,
    event;
    n_samples = 100
)
    p_post = plot(title = "Posterior Predictive Check",
        xlabel = "Time (days)",
        ylabel = "Survival Probability",
        legend = :bottomright)

    # Observed KM curve
    km_obs = fit(KaplanMeier, y, event)
    plot!(p_post, km_obs.events.time, km_obs.survival;
        seriestype = :steppost,
        label = "Observed Data",
        color = :black,
        linewidth = 2)

    params_chain = get(posterior_samples; section = :parameters)
    total_draws  = length(params_chain.α)

    for _ in 1:n_samples
        idx       = rand(1:total_draws)
        α_sample  = params_chain.α[idx]
        β_sample  = [params_chain.β[1][idx], params_chain.β[2][idx], params_chain.β[3][idx]]

        fixed_model = fix(model, (; α = α_sample, β = β_sample))
        y_sim       = fixed_model().y

        km_sim = fit(KaplanMeier, y_sim, event)
        plot!(p_post, km_sim.events.time, km_sim.survival;
            seriestype = :steppost,
            label = "",
            color = :lightblue,
            alpha = 0.3,
            xlims = (0, maximum(y) * 1.1))
    end

    display(p_post)
end

plot_posterior_pred_check_ll(model_ll, posterior_chains_ll, y, event)
```

### LOO and LOO‑PIT ECDF

```julia
using DynamicPPL, MCMCChains, ArviZ, DimensionalData

# Point‑wise log‑likelihoods
param_samples_ll = MCMCChains.get_sections(posterior_chains_ll, :parameters)
loglik_dict_ll   = DynamicPPL.pointwise_loglikelihoods(
    conditioned_model_ll,
    param_samples_ll
)

# Posterior predictive draws
post_pred_chain_ll = DynamicPPL.predict(model_ll, param_samples_ll)

# Reshape for ArviZ
n_draws, _, n_chains = size(posterior_chains_ll)
n_obs = length(y)
loglik_vals_ll   = Array{Float64}(undef, n_draws, n_chains, n_obs)
postpred_vals_ll = similar(loglik_vals_ll)

for i in 1:n_obs
    vn_str = "y[$i]"
    vn_sym = Symbol(vn_str)
    loglik_vals_ll[:, :, i]   .= loglik_dict_ll[vn_str]
    postpred_vals_ll[:, :, i] .= dropdims(post_pred_chain_ll[[vn_sym],].value, dims = 2)
end

idata_ll = ArviZ.from_mcmcchains(
    posterior_chains_ll;
    log_likelihood       = (; y = loglik_vals_ll),
    posterior_predictive = (; y = postpred_vals_ll),
    observed_data        = (; y = y),
    coords = (obs_id = 1:n_obs, coef_dim = 1:size(X, 2)),
    dims   = (y = [:obs_id], β = [:coef_dim])
)

# LOO and LOO‑PIT
loo_ll = ArviZ.loo(idata_ll)
display(loo_ll)

plot_loo_pit(idata_ll; y = "y", ecdf = true, color = "maroon")
gcf()
```