using Turing

@model function my_own_submodel(p)
    a ~ truncated(Normal(); lower=0)
    b ~ Gamma(p)
    return a + b
end

@model function my_model(data, p)
    x ~ to_submodel(my_own_submodel(p))
    data .~ Poisson(x)
end

model = my_model([10, 11, 8, 8, 12], 1.0)
chain = sample(model, NUTS(), 1000)  # This chain will have two variables: x.a and x.b
