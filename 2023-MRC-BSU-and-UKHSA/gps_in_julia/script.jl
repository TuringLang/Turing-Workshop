using AbstractGPs, CSV, DataDeps, DataFrames, KernelFunctions, LinearAlgebra,
    LogExpFunctions, MCMCChains, Plots, Random, Turing, Zygote

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
# 2. start from basics and build towards complete example (~45 mins)
#    - brief intro to GPs
#    - KernelFunctions.jl
#    - AbstractGPs.jl
#    - strategies for utilising GPs in Turing
# 3. free-form time to work on whatever is interesting (~35 mins)
#    - some curated / datasets will be provided
#    - you might also expand on anything you've found interesting thus far in the session

# We're going to use Zygote to perform AD.
Turing.setadbackend(:zygote)

# Very small dataset from Gelman et al.
register(DataDep(
    "putting",
    "Putting data from BDA",
    "http://www.stat.columbia.edu/~gelman/book/data/golf.dat",
    "fc28d83896af7094d765789714524d5a389532279b64902866574079c1a977cc",
));

@model function putting_model(d, d_pred, n, y)
    v ~ InverseGamma(2, 3)
    l ~ InverseGamma(2, 3)
    f = GP(v * with_lengthscale(SEKernel(), l))
    fx = f(d, 1e-4)
    f_latent ~ fx
    y ~ Product(Binomial.(n, logistic.(f_latent)))

    # Generated quantities.
    fx_pred = rand(posterior(fx, f_latent)(d_pred, 1e-4))
    return Zygote.dropgrad((fx_pred=fx_pred, p=logistic.(fx_pred)))
end

function putting_example()

    # Load up data.
    fname = joinpath(datadep"putting", "golf.dat")
    df = CSV.read(fname, DataFrame; delim=' ', ignorerepeated=true)

    # Construct a simple model with a latent GP.
    d_pred = 1:0.2:21
    model = putting_model(df.distance, d_pred, df.n, df.y)
    chn = sample(Xoshiro(123456), model, NUTS(), 100)

    # Plot probs of success.
    L = reduce(hcat, map(x -> x.p, generated_quantities(model, chn)))
    p = plot()
    plot!(d_pred, L; label="", color=:blue, alpha=0.2)
    scatter!(df.distance, df.y ./ df.n; label="", color=:red)
    savefig(p, "putting_success_probs.png")
end
