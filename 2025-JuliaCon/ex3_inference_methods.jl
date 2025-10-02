using Turing, StatsPlots

@model function linear_regression(x)
    # Priors
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    σ² ~ truncated(Normal(0, 1), lower=0)

    # Likelihood
    μ = α .+ β .* x
    y ~ MvNormal(μ, σ² * I)
end

x = rand(10)
y = 0.4 .* x .- 0.4 .+ randn(10)

m = linear_regression(x) | (; y=y)

# MCMC sampling with
# ... NUTS
sample(m, NUTS(), 1000)
# ... MH
sample(m, MH(), 1000)
# ... a Gibbs sampler that uses a different subsampler for each parameter
sampler = Gibbs(:α => HMC(0.1, 10), :β => ESS(), :σ² => MH())
sample(m, sampler, 1000)
# ... particle Gibbs
sample(m, PG(20), 1000)

# Variational inference
q_init = q_fullrank_gaussian(m)
vi(m, q_init, 1000)

# Mode estimation for maximum a posteriori
maximum_a_posteriori(m)
# or maximum likelihood
maximum_likelihood(m)
