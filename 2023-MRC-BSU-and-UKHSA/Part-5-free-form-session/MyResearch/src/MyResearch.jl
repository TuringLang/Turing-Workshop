module MyResearch

using DataFrames, CSV, Distributions

export NegativeBinomial2

function NegativeBinomial2(μ, ϕ)
    p = 1/(1 + μ/ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end

function load_data()
    return DataFrame(CSV.File(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "..",
            "Part-2-More-Julia-and-some-Bayesian-inference",
            "data",
            "influenza_england_1978_school.csv"
        )
    ))
end

end # module MyResearch
