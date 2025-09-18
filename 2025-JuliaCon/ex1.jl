using Turing

@model function f()
    x ~ Normal()
    y ~ Normal(x)
end

model = f() | (; y = 1.0)
chain = sample(model, NUTS(), 1000)
