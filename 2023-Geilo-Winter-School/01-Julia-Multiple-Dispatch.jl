# Example with One-hot Vector
# Based on: https://storopoli.github.io/Bayesian-Julia/pages/01_why_Julia/

# How to represent one-hot vectors?
# We actually just need two things:
# 1. Length
# 2. Which index is "hot"
import Base: size, getindex

struct OneHotVector <: AbstractVector{Int}
    len::Int
    ind::Int
end

size(v::OneHotVector) = (v.len,)

getindex(v::OneHotVector, i::Integer) = Int(i == v.ind)

# Since OneHotVector is a struct derived from AbstractVector we can use all of the methods
# previously defined for AbstractVector and it simply works right off the bat.
# Here we are constructing an Array with a list comprehension:
onehot = [OneHotVector(3, rand(1:3)) for _ = 1:4]

# Define a new function inner_sum that is basically a recursive dot product with a summation:
using LinearAlgebra

function inner_sum(A, vs)
    t = zero(eltype(A))
    for v in vs
        t += inner(v, A, v) # multiple dispatch!
    end
    return t
end

inner(v, A, w) = v ⋅ (A * w) # very general definition

# It works!
A = rand(3, 3)
vs = [rand(3) for _ = 1:4]
inner_sum(A, vs)

# Why?
supertype(OneHotVector) # subtype of AbstractVector

# But this is still slow:
using BenchmarkTools

@btime inner_sum($A, $onehot);

# We can optimize this.
# Multiplying a matrix by a one-hot is pretty much column indexing:
import Base: *

*(A::AbstractMatrix, v::OneHotVector) = A[:, v.ind]

# And doing inner on a matrix sandwiched between one-hot vectors is row/column indexing:
inner(v::OneHotVector, A, w::OneHotVector) = A[v.ind, w.ind]

# Now benchmark again:
@btime inner_sum($A, $onehot);
