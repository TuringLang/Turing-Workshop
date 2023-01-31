module GeiloWinterSchool2023Part2

using DiffEqBase
using Turing
using StatsPlots
using DataFrames
using SciMLSensitivity

using ReverseDiff: ReverseDiff

export to_chains, plot_trajectories!, plot_trajectories, plot_trajectory_quantiles!, plot_trajectory_quantiles, issuccess


# HACK: https://github.com/JuliaDiff/ReverseDiff.jl/pull/216
if isdefined(ReverseDiff, :supports_linear_indexing)
    ReverseDiff.supports_linear_indexing(::DiffEqBase.ODESolution) = true

    import DiffEqBase: solve_up
    ReverseDiff.@grad function solve_up(prob, sensealg, u0, p, args...; kwargs...)
        out = DiffEqBase._solve_adjoint(prob, sensealg, ReverseDiff.value(u0),
                                        ReverseDiff.value(p),
                                        SciMLBase.ReverseDiffOriginator(), args...; kwargs...)
        function actual_adjoint(_args...)
            original_adjoint = out[2](_args...)
            if isempty(args) # alg is missing
                tuple(original_adjoint[1:4]..., original_adjoint[6:end]...)
            else
                original_adjoint
            end
        end

        out[1], actual_adjoint
    end
end

# Improve `show` a bit for the sake of presentation and demonstration.
function Base.show(io::IO, model::DynamicPPL.Model)
    println(io, "Model(")
    print(io, "  args = ")
    println(io, keys(model.args))
    print(io, "  defaults = ")
    println(io, keys(model.defaults))
    print(io, "  context = ")
    println(io, model.context)
    print(io, ")")
end

# Utility method for converting a `Vector{<:NamedTuple}` to a `Chains` object.
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

# HACK: https://github.com/TuringLang/DynamicPPL.jl/pull/452
function DynamicPPL.TestUtils.varname_leaves(vn::DynamicPPL.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        lens = DynamicPPL.Setfield.PropertyLens{sym}()
        DynamicPPL.TestUtils.varname_leaves(vn ∘ lens, get(val, lens))
    end
    return Iterators.flatten(iter)
end

# Some plotting utilities.
"""
    plot_trajectories!(p::Plots.Plot, chain::MCMCChains.Chains; kwargs...)

Plot trajectories in `chain`.

# Keyword arguments
n = 1000, trajectory_color="#BBBBBB", alpha=1/√n
- `n`: number of trajectories to sample. Default: 1000.
- `trajectory_color`: color of the trajectories to use. Default: "#BBBBBB".
- `alpha`: alpha to use when plotting the trajectories. Default: `1/√n`.
- `data`: include the data in the plot. Default: `nothing`.
- `tspan`: 2-tuple containing first and last time represented by `chain`. Default: (1, 14)
"""
function plot_trajectories!(
    p::Plots.Plot, chain::MCMCChains.Chains;
    n = 1000, data=nothing, tspan=(1, 14),
    trajectory_color="#BBBBBB", alpha=1/√n
)
    # Convert the `chain` into a flattened `Array`.
    chain_arr = Array(chain; append_chains=true)  # Shape: [num_chains × num_iterations, num_params]
    # Sample a subset of indices to use.
    total_num_samples = size(chain_arr, 1)
    indices = rand(1:total_num_samples, n)  # Shape: [n, ]
    # Plot 'em!
    chain_arr = chain_arr[indices, :]  # Shape: [n, num_params]
    for in_bed in eachrow(chain_arr)
        plot!(p, tspan[1]:tspan[2], in_bed, alpha=alpha, color=trajectory_color)
    end

    if !isnothing(data)
        scatter!(p, 1:14, data.in_bed, label="Data", color="black")
    end

    return p
end

"""
    plot_trajectories(chain::MCMCChains.Chains; kwargs...)

Plot trajectories in `chain`.

See [`plot_trajectories!`](@ref) for more info.
"""
plot_trajectories(chain::MCMCChains.Chains; kwargs...) = plot_trajectories!(plot(), chain; kwargs...)

"""
    plot_trajectory_quantiles!(p::Plots.Plot, chain::MCMCChains.Chains; kwargs...)

Plot quantiles of trajectories in `chain`.

By default, the 95% quantiles are used.

# Keyword arguments
- q: quantiles to compute (only three ordered values supported). Default: `[0.025, 0.5, 0.975]`.
- `data`: if specified, include the data in the plot. Default: `nothing`.
- `tspan`: 2-tuple containing first and last time represented by `chain`. Default: (1, 14)
"""
function plot_trajectory_quantiles!(
    p::Plots.Plot, chain::MCMCChains.Chains;
    q=[0.025, 0.5, 0.975], data=nothing, tspan=(1, 14)
)
    # Compute quantiles.
    qs = Array(DataFrame(quantile(chain; q = q))[:, Not(:parameters)]);

    # Plot!
    num_timesteps = size(qs, 1)
    lb, mid, ub = qs[:, 1], qs[:, 2], qs[:, 3]
    plot!(p, tspan[1]:tspan[2], mid, ribbon=(mid - lb, ub - mid))

    if !isnothing(data)
        scatter!(p, 1:14, data.in_bed, label="Data", color="black")
    end

    return p
end


"""
    plot_trajectory_quantiles(chain::MCMCChains.Chains; kwargs...)

Plot quantiles of trajectories in `chain`.

See [`plot_trajectory_quantiles!`](@ref) for more info.
"""
plot_trajectory_quantiles(chain::MCMCChains.Chains; kwargs...) = plot_trajectory_quantiles!(plot(), chain; kwargs...)

# Utility method to check if we're happy with ODE solution.
"""
    issuccess(sol)

Return `true` if the ODE solution `sol` is successful, and `false` otherwise.
"""
function issuccess(sol)
    # Some AD frameworks use their custom types to wrap arrays.
    # Hence we just need to unwrap before we access the `retcode`.
    # Thanks to multiple dispatch, this is a no-op in the case where
    # there's no need for unwrapping.
    sol_unwrapped = DiffEqBase.value(sol)
    # `Terminated` can occur if we give the solver a callback-function which
    # dynamically decides whether or not to terminate. `Success` is, well, success.
    return (sol_unwrapped.retcode === ReturnCode.Success) || (sol_unwrapped.retcode === ReturnCode.Terminated)
end

# HACK: Don't need to diff through `issuccess`, so we just drop the tracking.
issuccess(sol::ReverseDiff.TrackedArray) = issuccess(ReverseDiff.value(sol))
issuccess(sol::AbstractArray{<:ReverseDiff.TrackedReal}) = issuccess(ReverseDiff.value.(sol))

end # module GeiloWinterSchool2023Part2
