using DataFrames, CSV

function load_data()
    return DataFrame(CSV.File("/home/tor/Projects/public/Turing-Workshop/2023-MRC-BSU-and-UKHSA/Part-2-More-Julia-and-some-Bayesian-inference/data/influenza_england_1978_school.csv"))
end
