using AbstractGPs, CSV, DataDeps, DataFrames, KernelFunctions, LinearAlgebra,
    LogExpFunctions, MCMCChains, Plots, Random, ReverseDiff, Turing

# We're going to use ReverseDiff to perform AD.
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Very small dataset from Gelman et al.
register(DataDep(
    "putting",
    "Putting data from BDA",
    "http://www.stat.columbia.edu/~gelman/book/data/golf.dat",
    "fc28d83896af7094d765789714524d5a389532279b64902866574079c1a977cc",
));

@model function putting_model(d, n; jitter=1e-4)
    v ~ Gamma(2, 1)
    l ~ Gamma(4, 1)
    f = GP(v * with_lengthscale(SEKernel(), l))
    f_latent ~ f(d, jitter)
    y ~ product_distribution(Binomial.(n, logistic.(f_latent)))
    return (fx=f(d, jitter), f_latent=f_latent, y=y)
end

function plot_data(d, n, y, xticks, yticks)
    ylims = (0, round(maximum(n), RoundUp; sigdigits=2))
    margin = -0.5 * Plots.mm
    plt = plot(xticks=xticks, yticks=yticks, ylims=ylims, margin=margin, grid=false)
    bar!(plt, d, n; color=:red, label="", alpha=0.5)
    bar!(plt, d, y; label="", color=:blue, alpha=0.7)
    return plt
end

function putting_example()

    # Load up data.
    fname = joinpath(datadep"putting", "golf.dat")
    df = CSV.read(fname, DataFrame; delim=' ', ignorerepeated=true)

    # Construct model and run some prior predictive checks.
    m = putting_model(Float64.(df.distance), df.n)
    hists = map(1:20) do j
        xticks = j > 15 ? :auto : nothing
        yticks = rem(j, 5) == 1 ? :auto : nothing
        return plot_data(df.distance, df.n, m().y, xticks, yticks)
    end
    savefig(plot(hists...; layout=(4, 5), dpi=300), "prior_pred.png")

    # Construct a simple model with a latent GP.
    m_post = m | (y=df.y, )
    chn = sample(Xoshiro(123456), m_post, NUTS(), 1_000)
    display(chn)
    println()

    # Compute sample probabilities of success.
    d_pred = 1:0.2:21
    samples = map(generated_quantities(m_post, chn)[1:10:end]) do x
        return logistic.(rand(posterior(x.fx, x.f_latent)(d_pred, 1e-4)))
    end
    p = plot()
    plot!(d_pred, reduce(hcat, samples); label="", color=:blue, alpha=0.2)
    scatter!(df.distance, df.y ./ df.n; label="", color=:red)
    savefig(p, "putting_success_probs.png")

    # Generate some replications of y to get a sense for the remaining marginal uncertainty.
    ys = [x.y for x in generated_quantities(m, predict(rng, m, chn))[1:10:end]]
    post_hists = map(1:20) do j
        xticks = j > 15 ? :auto : nothing
        yticks = rem(j, 5) == 1 ? :auto : nothing
        return plot_data(df.distance, df.n, ys[j], xticks, yticks)
    end
    hists = vcat(plot_data(df.distance, df.n, df.y, nothing, :auto), post_hists[2:end])
    savefig(plot(hists...; layout=(4, 5), dpi=300), "posterior_pred.png")

    # # Sample using ESS + HMC.
    # chn_ess = sample(Xoshiro(123456), m_post, Gibbs(ESS(:f_latent), HMC(0.1, 3, :v, :l)), 1_000)
end

# Notes:
# 5. Motivate GPs - chat with Tor / other examples from outside infectious diseases
# 11. check out time series example in the docs
