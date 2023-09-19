using AbstractGPs, CSV, DataDeps, DataFrames, KernelFunctions, LinearAlgebra,
    LogExpFunctions, MCMCChains, Plots, Random, ReverseDiff, Turing

# Aim of session: understand core tools available in Julia for GPs (how they are designed,
# how to extend them as required, how to pick + choose which bits you use in your particular
# application, and how to use them), and how they interact with Turing.jl.
#
# If you are interested in GPs, my hope is that you leave this session with an understanding
# of the tooling that's available to you in Julia, why it looks the way it does, its
# limitations, and how you might extend it to suit your needs.
#
# Regardless whether you are interested in GPs, my hope is that this session gets you
# further accumstomed to the kinds of abstractions and package designs that are commonly
# employed in Julia.
#
# Overall plan:
# 1. intro + (trivial) complete example (~10 mins)
# 2. start from basics and build towards complete example (~35 mins)
#    - brief intro to GPs
#    - KernelFunctions.jl
#    - AbstractGPs.jl
#    - strategies for utilising GPs in Turing
# 3. free-form time to work on whatever is interesting, and clarify and issues (~15 mins)
#    - you might also expand on anything you've found interesting thus far in the session

# We're going to use ReverseDiff to perform AD.
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function LinearAlgebra.:*(
    x::LowerTriangular{<:Number, <:ReverseDiff.TrackedMatrix{T, D}},
    y::ReverseDiff.TrackedVector{T, D},
) where {T, D}
    @show typeof(collect(x))
    return ReverseDiff.record_mul(x, y, D)
end

function noncentered(fx::AbstractMvNormal)
    m, C = mean_and_cov(fx)
    b = Bijectors.Shift(m) ∘ Bijectors.Scale(cholesky(Symmetric(C)).L)
    return transformed(MvNormal(ones(length(fx))), b)
end

# Very small dataset from Gelman et al.
register(DataDep(
    "putting",
    "Putting data from BDA",
    "http://www.stat.columbia.edu/~gelman/book/data/golf.dat",
    "fc28d83896af7094d765789714524d5a389532279b64902866574079c1a977cc",
));

@model function putting_model(d, n; jitter=1e-4)
    v ~ InverseGamma(2, 3)
    l ~ InverseGamma(2, 3)
    f = GP(v * with_lengthscale(SEKernel(), l))
    f_latent ~ f(d, jitter)
    y_dist = product_distribution(Binomial.(n, logistic.(f_latent)))
    y ~ y_dist
    return (fx=f(d, jitter), f_latent=f_latent, y=y, y_dist=y_dist)
end

function putting_example()

    # Load up data.
    fname = joinpath(datadep"putting", "golf.dat")
    df = CSV.read(fname, DataFrame; delim=' ', ignorerepeated=true)

    # Construct model and run some prior predictive checks.
    m = putting_model(Float64.(df.distance), df.n)
    hists = [bar(df.distance, m().y; label="", xlabel="") for _ in 1:20]
    savefig(plot(hists...; layout=(4, 5), dpi=300), "prior_pred.png")

    # Construct a simple model with a latent GP.
    m_post = m | (y=df.y, )
    chn = sample(Xoshiro(123456), m_post, NUTS(), 1_000)

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
    ys = [rand(x.y_dist) for x in generated_quantities(m_post, chn)[1:10:end]]
    hists = vcat(
        bar(df.distance, df.y; label="", xlabel=""),
        [bar(df.distance, ys[j]; label="", xlabel="") for j in 1:8],
    )
    savefig(plot(hists...; layout=(3, 3), dpi=300), "posterior_pred.png")

    # Sample using ESS + HMC.
    chn_ess = sample(Xoshiro(123456), m_post, Gibbs(ESS(:f_latent), HMC(0.1, 3, :v, :l)), 1_000)
end

# Notes:
# 5. Motivate GPs - chat with Tor / other examples from outside infectious diseases
