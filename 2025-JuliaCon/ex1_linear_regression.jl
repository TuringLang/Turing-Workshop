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

m = linear_regression(x)
m_conditioned = m | (; y=y)

chain = sample(m_conditioned, NUTS(), 1000)
plot(chain)
