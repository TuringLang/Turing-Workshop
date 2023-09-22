using AbstractGPs, CSV, DataDeps, DataFrames, KernelFunctions, LinearAlgebra,
    LogExpFunctions, MCMCChains, Plots, Random, RDatasets, Turing, Zygote

@model function mcycle_model(t; jitter=1e-4)

    # Log variance process.
    v2 ~ Gamma(2, 1)
    l2 ~ Gamma(4, 1)
    log_σ = GP(v2 * with_lengthscale(SEKernel(), l2))
    log_σ_t ~ log_σ(t, jitter)

    # Mean process.
    v1 ~ Gamma(2, 1)
    l1 ~ Gamma(4, 1)
    f = GP(v1 * with_lengthscale(SEKernel(), l1))
    dist_y = transformed(f(t, 1e-2), Bijectors.Scale(exp.(log_σ_t)))
    y ~ dist_y
    return (y=y, dist_y=dist_y, f=f, log_σ_t=log_σ_t)
end

Turing.setadbackend(:zygote)

function main()
    df = dataset("MASS", "mcycle")
    m = mcycle_model(df.Times)

    scatter(df.Times, m().y)

    # Standardise the data.
    a = (df.Accel .- mean(df.Accel)) ./ std(df.Accel)

    m_post = m | (y=a, )
    sampler = Gibbs(ESS(:log_σ_t), HMC(0.03, 5, :v1, :l1, :v2, :l2))
    chn_ess = sample(Xoshiro(123456), m_post, sampler, 1_000)

    # Debug hypers.
    hypers = map(collect ∘ vec, get(chn_ess, [:l1, :v1, :l2, :v2]))
    savefig(plot(map(x -> plot(x; label=""), hypers)...; layout=(2, 2)), "hypers.png")

    # Make predictions for y.
    preds = predict(Xoshiro(123456), m, chn_ess)
    ys = [x.y for x in generated_quantities(m, preds)]
    plt = plot()
    plot!(plt, df.Times, ys[500:10:end]; alpha=0.5, label="", color=:blue)
    scatter!(df.Times, a; label="y", color=:red)
    savefig(plt, "mcycle_partially_conditioned_samples.png")

    # TODO: generate posterior samples at these locations and plot them
    t_pred = range(0.0, 60.0; length=250)
end
