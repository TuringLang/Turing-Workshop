using AbstractGPs, KernelFunctions, Random
import KernelFunctions: kernelmatrix, kernelmatrix_diag, TestUtils



#
# Using kernels
#

# You can call a kernel on individual inputs to return a scalar:
SEKernel()(5.0, 4.0)

# In practice, you usually want to compute the kernel for a collection of inputs.
# The function `kernelmatrix` does this.
# There are several methods, but the first two listed are the important ones:
# kernelmatrix(::Kernel, ::AbstractVector)
# and
# kernelmatrix(::Kernel, ::AbstractVector, ::AbstractVector)
# Take a look at the docstring for details:
@doc kernelmatrix

# For example, in the one-dimensional case, you might do something like:
x = randn(5)
y = randn(4)
kernelmatrix(SEKernel(), x, y)
kernelmatrix(SEKernel(), x)

# If your data is multi-dimensional, store it data in a `Matrix`, and wrap this in
# a `ColVecs` (if your matrix represents a collection of column vectors), and `RowVecs` (if
# your matrix represents a collection of row vectors). For example, if we have 3-dimensional
# data, we might do something like:
x = ColVecs(randn(3, 5))
y = ColVecs(randn(3, 4))
kernelmatrix(SEKernel(), x, y)
kernelmatrix(SEKernel(), x)

# Note that length(x) == 5 and length(y) == 4 -- `ColVecs` is itself an `AbstractVector`.
# It is a lightweight abstraction around a `Matrix` that says "interpret this data as a
# vector-of-vectors".

# This is rather than storing data as a vector-of-vectors, which has a number of negative
# performance implications.

# Note that the use of `ColVecs` and `RowVecs` resolves any ambiguity around how a matrix of
# data ought to be interpreted (i.e. is each column an input, or each row an input?), and
# doesn't introduce any performance problems. See the docs for more details around this
# design choice: 
# https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/design/#why_abstract_vectors


#
# Defining your own kernel
#

# A kernel is defined via a struct.
struct MyEQKernel{Tl<:Real, Tv<:Real} <: Kernel
    l::Tl
    v::Tv
    function MyEQKernel(l::Tl, v::Tv) where {Tl, Tv}
        @assert l > 0
        @assert v > 0
        return new{Tl, Tv}(l, v)
    end
end

# At a minimum, you must specify what evaluating your kernel on a pair of inputs returns.
(k::MyEQKernel)(x::Real, y::Real) = k.v * exp(-(x - y)^2 / (2 * k.l^2))

# To check that you have satisfied the `Kernel` interface, you can call the
# `TestUtils.test_interface` function.
rng = Xoshiro(123456)
TestUtils.test_interface(MyEQKernel(5.0, 4.0), randn(rng, 5), randn(rng, 5), randn(rng, 4))

# Specifying how to evaluate your kernel means that various high-level functions get default
# definitions: `kernelmatrix`, `kernelmatrix_diag`, and their in-place versions,
# `kernelmatrix`, and `kernelmatrix_diag!`.
# Often you will find it beneficial to performance to implement these directly, in addition
# to defining kernel evaluation:
kernelmatrix(k::MyEQKernel, x::Vector{<:Real}) = k.v * exp.((x .- x').^2 ./ (-2 * k.l))

# `TestUtils.test_interface` should be run again:
TestUtils.test_interface(MyEQKernel(5.0, 4.0), randn(rng, 5), randn(rng, 5), randn(rng, 4))

# `TestUtils.test_interface` has found a disagreement between the implementation of
# `kernelmatrix`, and `(k::MyEQKernel)(x, y)`. If we fix this, tests should pass.
kernelmatrix(k::MyEQKernel, x::Vector{<:Real}) = k.v * exp.((x .- x').^2 ./ (-2 * k.l^2))
TestUtils.test_interface(MyEQKernel(5.0, 4.0), randn(rng, 5), randn(rng, 5), randn(rng, 4))


#
# Using AbstractGPs.jl
#

# AbstractGPs provides tools which allow you to specify a GP,
# produce a multivariate Gaussian from the GP at a finite collection of inputs,  and
# do all the usual things you would hope to do with a multivariate Gaussian + conditioning
# on data to produce an updated GP.

# The GP type is a thin wrapper around a `MeanFunction` and `Kernel`. If you want a zero-
# mean GP, do not provide a `MeanFunction`:
f = GP(MyEQKernel(5.0, 4.0))

# This particular GP should be thought of as a distribution over functions.
# In practice we typically work with the multivariate Gaussian obtained by considering the
# GP at only a finite number of inputs:
x = randn(5)
fx = f(x, 0.1)

# In practice, we always add some noise (for numerical reasons). So `fx` should be thought
# of as the sum of `f` at `x` with a vector of `length(x)` IID Gaussian random variables,
# with zero-mean and variance `0.1`.

# `fx` is a multivariate Gaussian distribution, so you can sample from it:
y = rand(rng, fx)

# Compute the `logpdf` of a sample:
logpdf(fx, y)

# This function can be differentiated using Zygote, so is well-suited for use with Turing,
# which requires this for HMC.

# Moreover, you can condition on data:
f_post = posterior(fx, y)

# `f_post` is just another `AbstractGP`, and can do all of the things that a `GP` can.
