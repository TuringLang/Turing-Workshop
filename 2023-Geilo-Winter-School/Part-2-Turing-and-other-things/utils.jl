# Pkg.add(url="https://github.com/torfjelde/ReverseDiff.jl", rev="torfjelde/sort-of-support-non-linear-indexing")

using ReverseDiff: ReverseDiff

if isdefined(ReverseDiff, :supports_linear_indexing)
    using DiffEqBase
    ReverseDiff.supports_linear_indexing(::DiffEqBase.ODESolution) = true
end

# Pkg.add(url="https://github.com/TuringLang/DistributionsAD.jl.git", rev="torfjelde/lazy-array-perf")

using Turing

function unravel(x::NamedTuple, vns=Iterators.map(DynamicPPL.VarName, keys(x)))
    vn_iter = Iterators.flatten(DynamicPPL.TestUtils.varname_leaves(vn, get(x, vn)) for vn in vns)
    return vn_iter, Iterators.map(Base.Fix1(get, x), vn_iter)
end

"""
    to_chains(results; exclude, include)

Return a `MCMCChains.Chains` constructed from `results`.

If `exclude` is specified, then those keys will be excluded.
If `include` is specified, then those keys will be included.
"""
function to_chains(results::AbstractVector{<:NamedTuple}; exclude=nothing, include=nothing)
    @assert !isempty(results) "input is empty"

    # TODO: Handle ragged arrays.
    # Probably best done my first just mapping everything to flatten dictionaries, e.g.
    #
    #     x_vns, x_vals = unravel(x)
    #     OrderedDict(zip(map(Symbol, x_vns), x_vals))
    #
    # (or using Dictionaries.jl for better perf), and then basically `hcat`ing these
    # and insertin `missing` where necessary.

    example = first(results)

    # Construct names once.
    syms = collect(keys(example))
    # Filter out if desired.
    if !isnothing(include)
        syms = filter(∈(include), syms)
    end
    if !isnothing(exclude)
        syms = filter(∉(exclude), syms)
    end

    @assert !isempty(syms) "symbols to include is empty"
    
    # Convert to `VarName`.
    names = collect(first(unravel(example, map(DynamicPPL.VarName, syms))))

    # Extract the values.
    vals = mapreduce(hcat, results) do x
        # NOTE: Providing the `names` here assumes every sample has the same variables.
        collect(last(unravel(x, names)))
    end

    @assert !isempty(vals) "resulting array is empty"

    vals_transposed = transpose(vals)
    # Shape: iterations × num_vars × num_chains
    chain_array = reshape(vals_transposed, size(vals_transposed)..., 1)
    return MCMCChains.Chains(chain_array, names)
end

function to_chains(results::AbstractMatrix{<:NamedTuple}; kwargs...)
    return reduce(MCMCChains.chainscat, map(col -> to_chains(col; kwargs...), eachcol(results));)
end

# TODO: Make PR to DPPL.
function DynamicPPL.TestUtils.varname_leaves(vn::DynamicPPL.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        lens = DynamicPPL.Setfield.PropertyLens{sym}()
        DynamicPPL.TestUtils.varname_leaves(vn ∘ lens, get(val, lens))
    end
    return Iterators.flatten(iter)
end

# to_chains([(x = 1, y = [2,2], z = (w = [5],)), (x = 3, y = [4,4], z = (w = [6],))])
