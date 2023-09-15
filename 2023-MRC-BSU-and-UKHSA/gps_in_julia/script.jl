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
# 2. start from basics and build towards complete example (~35 mins)
#    - brief intro to GPs
#    - KernelFunctions.jl
#    - AbstractGPs.jl
#    - strategies for utilising GPs in Turing
# 3. free-form time to work on whatever is interesting, and clarify and issues (~15 mins)
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

@model function putting_model(d, n)
    v ~ InverseGamma(2, 3)
    l ~ InverseGamma(2, 3)
    f = GP(v * with_lengthscale(SEKernel(), l))
    fx = f(d, 1e-4)
    f_latent ~ fx
    y ~ Product(Binomial.(n, logistic.(f_latent)))
    return (fx=fx, f_latent=f_latent)
end

function putting_example()

    # Load up data.
    fname = joinpath(datadep"putting", "golf.dat")
    df = CSV.read(fname, DataFrame; delim=' ', ignorerepeated=true)

    # Construct a simple model with a latent GP.
    d_pred = 1:0.2:21
    model = putting_model(df.distance, df.n) | (y=df.y, )
    chn = sample(Xoshiro(123456), model, NUTS(), 100)

    # Plot probs of success.
    samples = map(generated_quantities(model, chn)) do x
        return logistic.(rand(posterior(x.fx, x.f_latent)(d_pred, 1e-4)))
    end
    p = plot()
    plot!(d_pred, reduce(hcat, samples); label="", color=:blue, alpha=0.2)
    scatter!(df.distance, df.y ./ df.n; label="", color=:red)
    savefig(p, "putting_success_probs.png")
end

# Notes:
# 6. How to use block gibbs?
# 7. When do you not want to use these abstractions? e.g. fixed kernel matrix, variable length scale across all dims.
