module TorsWorkshop

using Turing: Turing, DynamicPPL

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


end # module TorsWorkshop
