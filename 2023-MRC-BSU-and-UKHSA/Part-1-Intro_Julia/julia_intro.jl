# Julia Cheat Sheet: https://cheatsheet.juliadocs.org/

###
### Basic Data Types in Julia
###

# Integer
x = 1

# Float
y = 2.0

# unicode variable names: \alpha<TAB>
α = 1
β = 2
γ = 3
x₁ = 1 # x\_1<TAB>
x⁵ = 5 # don't do this
🐘 = "elephant" # \:elephant:
🦓 = "zebra" # \:zebra_face:
🦁 = "lion" # \:lion_face: # definitely don't do this
🦁 * ": ROAR!!"

# Boolean
b = true
b = false # all lower case

# Strings
s = "string"

###
### Arrays in Julia
###

# Vector
a = [1, 2, 3]
a[1]
a[begin]
a[end]
a[0] # error!: 1-based indexing
a[-1] # error!: no negative indexing
a[1:2] == [1, 2] # including whole range, unlike Python

# Matrices
m = [1 2 3; 4 5 6; 7 8 9]  # Declare a matrix
m[1, 1]  # Accessing elements in a matrix
m[1, :]  # Accessing rows in a matrix
m[1] # linear indexing
m[:, 2]
m[4:6] # column-major order

sub_matrix = m[1:2, 1:2] # indexing copies
sub_matrix[1, 1] = 10
m[1, 1] == 10

sub_matrix_view = view(m, 1:2, 1:2) # view does not
sub_matrix_view[1, 1] = 10
m[1, 1] == 10
setindex!(m, 1, 1, 1)

# Array Functions
length(a)  # Get the number of elements in an array
size(m)  # Get the size of an array
ndims(m)  # Get the number of dimensions of an array
vcat(m...) # Concatenate arrays vertically
hcat(m...) # Concatenate arrays horizontally
push!(a, 4) # Add an element to the end of an array
pop!(a) # Remove the last element from an array
a = reverse(a) # Reverse the order of elements in vector a
sort(a) # Sort an array in ascending order
sort!(a) # Sort an array in place
maximum(a) # Get the maximum value in an array
minimum(a) # Get the minimum value in an array
sum(a) # Get the sum of all elements in an array
prod(a) # Get the product of all elements in an array

# Vector, Matrix is alias for Array type
typeof(a)
typeof(m)

###
### Functions and Control Structures in Julia
###

# Basic Function Form
function increment(x)
    return x + 1
end

# Control Structures

# Example of if statement
if x > y
    println("x is greater than y")
elseif x < y
    println("x is less than y")
else
    println("x is equal to y")
end

# Example of for loop
for i in a
    println(i)
end

# Example of while loop
i = 1
while i <= 3
    println(i)
    i += 1
end

# Example of break/continue
for i in a
    if i == 2
        continue
    elseif i == 3
        break
    end
    println(i)
end

# Short-circuit evaluation
true && println("true")

# Ternary operator
true ? println("true") : println("false")

x = 1; y = 2.0 # multiple expressions on one line

# Function Continued

# Alternative function definition syntax: just like math equations
increment(x) = x + 1
increment(y)

# Basic Arithmetic Operators (build-in functions)
x + y
x - y
x * y
x / y
x ^ y
x % y
inv(y)
sqrt(y)
sin(y)
atan(y)
log(y)
2x + 3y # can omit * for multiplication

# Dictionary in Julia
d = Dict("name" => "Julia", :version => 1.10)
d["name"]  # Accessing elements in a dictionary
d[:version]
:version isa Symbol

# Tuples in Julia
t = (1, "two", 3.0)
t[1]
getindex(t, 2)

# NamedTuples in Julia
nt = (name = "Julia", version = 1.10)
nt.name  # Accessing elements in a NamedTuple
nt[:version]

# x, y different type: multiple dispatch and primitive types
@which x + x
@which y + y
@which x + y
x + y * im # complex number
x isa Real 
x isa Number
x + y * im isa Number

# Define function that specifies argument types -- taste of multiple dispatch
function increment(a::Int)
    return a + 1 
end

function increment(a::Float64)
    # return a + 2
    a + 2 # last line is returned if no explicit return
end

increment(1) == 2
increment(1.0) == 3.0

# Mutation on the Function theme

# Keyword arguments
function add(x; a, b = 2)
    return x + a + b
end

add(1; a = 2)
add(1; a = 2, b = 3)

# Optional arguments
function add(x, a = 1, b = 2)
    return x + a + b
end

add(1, 2, 3)

# Varargs
function sum_values(x, a...)
    println(a)
    return x + sum(a)
end

sum_values(1, 2, 3, 4, 5)

# Keyword varargs
function sum_values(x; a...)
    println(a)
    return x + sum(values(a))
end

sum_values(1; a = 2, b = 3, c = 4)

# Example of broadcasting
a = [1, 2, 3]
b = a .+ 2
c = ones(Int, 3)
c .= a .+ 2
c == b

# Example of comprehensions
[x^2 for x in a]
[x^2 for x in a if x > 1]

# Anonymous functions (lambda functions)
(x -> x + 1)(x) == 2

# Example of map function, also fold/reduce, mapreduce ... 
map(x -> x * 2, a)

# Various do block
map([1, 2, 3]) do x # same as map(x -> x, [1, 2, 3])
    x * 2
end
# Various do block
map([1, 2, 3]) do x # same as map(x -> x, [1, 2, 3])
    a = x * 2
    return a
end

# Use tempname() to create a temporary file
filename = tempname()
open(filename, "w") do io
    write(io, "Hello world")
end
read(filename) # returns a byte array
read(filename, String) # same as above

# Try/Catch/Finally
try
    println("Trying to read a file...")
    open("non_existent_file.txt", "r")
catch e
    println("Caught an error: ", e)
finally
    println("This finally block gets executed no matter what!")
end

# struct and multiple dispatch
abstract type Animal end
abstract type Herbivore <: Animal end
abstract type Carnivore <: Animal end

struct Elephant <: Herbivore
    name::String
    daily_food_cost::Float64
end

struct Zebra <: Herbivore
    name::String
    daily_food_cost::Float64
end

mutable struct Lion <: Carnivore
    name::String
    lions_share::Float64
end

# compare these to Python's @property
function get_daily_food_cost(animal::Animal)
    return animal.daily_food_cost
end

function get_daily_food_cost(animal::Lion)
    return animal.lions_share
end

# compare these to Python's @abstractmethod
function daily_diet(animal::Animal)
    return "The daily diet details are not available."
end

function daily_diet(animal::Herbivore)
    return "$(animal.name) is a $(typeof(animal)), it primarily eats plants."
end

function daily_diet(animal::Carnivore)
    return "The animal is a carnivore, it primarily eats meat."
end

function daily_diet(animal::Elephant)
    return "The elephant primarily eats grass, fruits, and roots. Daily food cost: $(get_daily_food_cost(animal)) USD."
end

function daily_diet(animal::Lion)
    return "The lion primarily eats large ungulates such as zebras and wildebeests. Daily food cost: $(get_daily_food_cost(animal)) USD."
end

function yearly_food_cost(animal::Herbivore)
    summer_cost = get_daily_food_cost(animal) * 120
    winter_cost = get_daily_food_cost(animal) * 1.5 * 240
    return "Yearly food cost: $(summer_cost + winter_cost) USD."
end

function yearly_food_cost(animal::Carnivore)
    return "Yearly food cost: $(get_daily_food_cost(animal) * 365) USD."
end

elephant = Elephant("Dumbo", 50.0)
zebra = Zebra("Marty", 25.0)
lion = Lion("simba", 100.0)
lion.name = "SIMBA" # mutable struct

println(daily_diet(elephant))
println(daily_diet(lion))
println(daily_diet(zebra))

println(yearly_food_cost(elephant))
println(yearly_food_cost(lion))
