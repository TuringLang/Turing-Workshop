# The story of a little Norwegian boy

There once was a little Norwegian boy

![img](.notes/attachments/A_litle_Norwegian_boy/2023-01-18_14-49-24_471337_3317365246956_1262712540_o.jpg)

When this little boy was 20 years old, he was working as a parking guard near Preikestolen/Pulpit rock

![img](.notes/attachments/A_litle_Norwegian_boy/2023-01-18_14-57-08_Preikestolen-plateau-Go-Fjords-Bob-Engelsen-P1026771_kljg5o.jpeg)

One day when there was it was raining and there was no one hiking, the little boy had an amazing idea

> Maybe I can use this method of Mr. Bayes I learned a bit about yesteday to model Premier League?

The little boy got very excited and started looking for stuff on the big interwebs

The little boy came across this

![img](.notes/attachments/A_litle_Norwegian_boy/2023-01-18_14-46-02_Screenshot_20230118_144454.png)

And got <span class="underline">very</span> excited

But at the time, the little boy knew next to <span class="underline">nothing</span> about programming

The little boy couldn't write the code to do the inference

Whence the little boy became a <span class="underline">sad</span> little boy :(

But time heals all wounds, and at some point the little boy learned Python

And in Python, the boy found the *probabilistic programming language* `pymc3`

> Maybe I can use `pymc3` to perform inference in that Premier League model?

And so the sad boy once more became an excited little boy :)

But there was a problem

The boy wanted to write a for-loop in his model, but the model didn't want it to be so and complained!

The boy got frustrated and gave up, once more becoming a sad little boy :(

The boy should have known that the computational backend `theano` that was used by `pymc3` at the time couldn't handle for-loop, and instead he should have used `scan`. But the boy was only 20-something years old; he didn't know.

Some years later the boy discovers a programming language called <span class="underline">Julia</span>

<div class="fragment (appear)">

Julia makes a few promises

1.  It's fast. Like *really* fast.
2.  It's interactive; doesn't require full compilation for you to play with it.
3.  It's (sort of) dynamically typed, so you don't have to specify types everywhere.

</div>

<div class="fragment (appear)">

The boy thinks

> Wait, but this sounds like Python but the only difference is that&#x2026;I CAN WRITE FOR-LOOPS WITHOUT FEELING BAD ABOUT IT?!

Yes, yes he could

And 3.5 years later, he's still writing for-loops. Well, sort of.

</div>


## But it really is fast

![img](.notes/attachments/A_litle_Norwegian_boy/2023-01-18_15-31-28_Screenshot_20230118_153122.png "<https://julialang.org/benchmarks/> (2023-01-18)")

And the consequences are

-   Even a naive implementation will be fast; and if you want to go faster, you just optimize the code *in Julia*!
    -   No need to drop down to C(++)
-   ⟹ "Every" package is written in Julia, so when you encounter a bug you can more easily debug because *the code is in the same language as you're writing code in*!
-   ⟹ Same for *extending* packages!
    -   Moreover, thanks to multiple dispatch, you can change a function *on the fly* to experiment with code you don't even own!


# Why Turing.jl?

The main benefit of Turing.jl is of course that **it's just Julia**.


## But Why Julia?

Fair enough.

-   It goes vroom, vroom!
-   


# Set up

```julia
include("utils.jl")
```

    Matrix (generic function with 1 method)


## Load into Julia

```julia
# Load the dataframe.
using Dates
using DataFrames, CSV

N = 763
data = DataFrame(CSV.File("data/influenza_england_1978_school.csv"));
print(data)
```

    14×4 DataFrame
     Row │ Column1  date        in_bed  convalescent 
         │ Int64    Date        Int64   Int64        
    ─────┼───────────────────────────────────────────
       1 │       1  1978-01-22       3             0
       2 │       2  1978-01-23       8             0
       3 │       3  1978-01-24      26             0
       4 │       4  1978-01-25      76             0
       5 │       5  1978-01-26     225             9
       6 │       6  1978-01-27     298            17
       7 │       7  1978-01-28     258           105
       8 │       8  1978-01-29     233           162
       9 │       9  1978-01-30     189           176
      10 │      10  1978-01-31     128           166
      11 │      11  1978-02-01      68           150
      12 │      12  1978-02-02      29            85
      13 │      13  1978-02-03      14            47
      14 │      14  1978-02-04       4            20

Notice that each of the columns have associated types

Let's visualize the samples:

```julia
using StatsPlots
pyplot()  # TODO: Overload show with the corresponding `MIME` for GR instead
```

    Plots.PyPlotBackend()

```julia
@df data scatter(:date, :in_bed, label=nothing, ylabel="Number of students in bed")
```

![img](./.ob-jupyter/40ec3a382b6a8be7d42c862e3d31d11596326ae9.png)


# Differential equations

Suppose we have some function \(f\) which describes how a state \(x\) evolves wrt. \(t\)

\begin{equation*}
\dv{x}{t} = f(x, t)
\end{equation*}

which we then need to integrate to obtain the actual state at some time \(t\)

\begin{equation*}
x(t) = \int_{0}^{t} \dv{x}{t} \dd{t} = \int_{0}^{t} f(x, t) \dd{t}
\end{equation*}

In many interesting scenarios numerical methods are required to obtain \(x(t)\)


## In Julia

Everything related to differential equations is provided by `DifferentialEquations.jl`

And I really do mean [*everything*](https://docs.sciml.ai/DiffEqDocs/stable/)

<div class="side-by-side">

![img](.notes/attachments/Differential_equations/2023-01-19_19-48-23_Screenshot_20230119_194737.png)

![img](.notes/attachments/Differential_equations/2023-01-19_19-48-41_Screenshot_20230119_194838.png)

</div>


## Example: SIR model

One particular example of an (ordinary) differential equation that you might have seen recently is the **SIR model** used in epidemiology

![img](.notes/attachments/Differential_equations/2023-01-19_19-56-00_sir_illu.png "<https://covid19.uclaml.org/model.html> (2023-01-19)")

The temporal dynamics of the sizes of each of the compartments are governed by the following system of ODEs:

\begin{equation*}
\begin{split}
  \dv{S}{t} &= - \beta S \frac{I}{N} \\
  \dv{I}{t} &= \beta S \frac{I}{N} - \gamma I \\
  \dv{R}{t} &= \gamma I
\end{split}
\end{equation*}

where

-   \(S(t)\) is the number of people susceptible to becoming infected (no immunity),
-   \(I(t)\) is the number of people currently infected (and infectious),
-   \(R(t)\) is the number of recovered people (we assume they remain immune indefinitely),
-   \(β\) is the constant rate of infectious contact between people,
-   \(\gamma\) the constant recovery rate of infected individuals

Converting this ODE into code is just

```julia
using DifferentialEquations

function SIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    N = 763  # population
    S, I, R = u
    β, γ = p

    du[1] = dS = -β * I * S / N
    du[2] = dI = β * I * S / N - γ * I
    du[3] = dR = γ * I
end
```

    SIR! (generic function with 1 method)

Not too bad!

```julia
# Figure out the timespan we're working with.
last(data.date) - first(data.date) + Day(1) # add 1 since we have an observation for every day
```

    14 days

That is, we have 14 days of observations.

```julia
# Include 0 because that's the initial condition before any observations.
tspan = (0.0, 14.0)

# Initial conditions are:
#   S(0) = N - 1; I(0) = 1; R(0) = 0
u0 = [N - 1, 1, 0.0]
```

    3-element Vector{Float64}:
     762.0
       1.0
       0.0

Now we just need to define the overall problem and we can solve:

```julia
# Just to check that everything works, we'll just use some "totally random" values for β and γ:
problem = let β = 2.0, γ = 0.6
    ODEProblem(SIR!, u0, tspan, (β, γ))
end
```

    ODEProblem with uType Vector{Float64} and tType Float64. In-place: true
    timespan: (0.0, 14.0)
    u0: 3-element Vector{Float64}:
     762.0
       1.0
       0.0

Aaaand

```julia
sol = solve(problem)
```

    retcode: Success
    Interpolation: specialized 4th order "free" interpolation, specialized 2nd order "free" stiffness-aware interpolation
    t: 23-element Vector{Float64}:
      0.0
      0.0023558376404244326
      0.025914214044668756
      0.11176872871946908
      0.26714420676761075
      0.47653584778586056
      0.7436981238065388
      1.0701182881347182
      1.4556696154809898
      1.8994815718103506
      2.4015425820305163
      2.9657488203418048
      3.6046024613854746
      4.325611232479916
      5.234036476235002
      6.073132270491685
      7.323851265223563
      8.23100744184026
      9.66046960467715
     11.027717843180652
     12.506967592177675
     13.98890399536329
     14.0
    u: 23-element Vector{Vector{Float64}}:
     [762.0, 1.0, 0.0]
     [761.9952867607622, 1.003297407481751, 0.001415831756055325]
     [761.9472927630898, 1.036873767352754, 0.015833469557440357]
     [761.7584189579304, 1.1690001128296739, 0.0725809292398516]
     [761.353498610305, 1.4522140137552049, 0.19428737593979384]
     [760.6490369821046, 1.9447820690728455, 0.4061809488225752]
     [759.3950815454128, 2.8210768113583082, 0.7838416432288186]
     [757.0795798160242, 4.437564277195732, 1.4828559067800167]
     [752.6094742865345, 7.552145919430467, 2.8383797940350495]
     [743.573784947305, 13.823077731564027, 5.603137321131049]
     [724.5575481927715, 26.909267078762316, 11.533184728466205]
     [683.6474029897502, 54.51612001957392, 24.836476990675976]
     [598.1841629858786, 109.41164143668018, 55.40419557744127]
     [450.08652743810205, 192.396449154863, 120.51702340703504]
     [259.11626253270623, 256.9925778114915, 246.89115965580237]
     [148.3573731526537, 240.10301213899098, 374.53961470835543]
     [76.52998017846475, 160.6373332952353, 525.8326865263001]
     [55.70519994004921, 108.7634182279299, 598.531381832021]
     [41.39587834423381, 55.09512088924873, 666.5090007665176]
     [35.87067243374374, 27.821838135708532, 699.3074894305479]
     [33.252184333490774, 13.087185981359177, 716.6606296851502]
     [32.08996839417716, 6.105264616193066, 724.8047669896299]
     [32.08428686823946, 6.070415830241046, 724.8452973015196]

We didn't specify a solver

DifferentialEquations.jl uses `AutoTsit5(Rosenbrock32())` by default

Which is a composition between

-   `Tsit5` (4th order Runge-Kutta), and
-   `Rosenbrock32` (3rd order stiff solver)

with automatic switching between the two

`AutoTsit5(Rosenbrock32())` covers many use-cases well, but see

-   <https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/>
-   <https://www.stochasticlifestyle.com/comparison-differential-equation-solver-suites-matlab-r-julia-python-c-fortran/>

for more info on choosing a solver

But this is the resulting solution

```julia
plot(
    sol,
    linewidth=2, xaxis="Time in days", label=["Suspectible" "Infected" "Recovered"],
    alpha=0.5, size=(500, 300)
)
scatter!(1:14, data.in_bed, label="Data", color="black")
```

![img](./.ob-jupyter/cd1bb1a21d3412fd689378b07ab893406ba0e463.png)

Doesn't really match the data; let's do better

We can also find the optimal values for \(\beta\) and \(\gamma\) by just minimizing some loss, e.g. sum-of-squares

\begin{equation*}
\ell(\beta, \gamma) = \sum_{i = 1}^{14} \bigg( F(u_0, t_i;\ \beta, \gamma) - y_i \bigg)^2
\end{equation*}

where \(\big( y_i \big)_{i = 1}^{14}\) are the observations, \(F\) is the integrated system

First we define the loss

```julia
# Define the loss function.
function loss(problem, p)
    # `remake` just, well, remakes the `problem` with `p` replaced.
    problem = remake(problem, p=p)
    # To ensure we get solutions _exactly_ at the timesteps of interest,
    # i.e. every day we have observations, we use `saveat=1` to tell `solve`
    # to save at every timestep (which is one day).
    sol = solve(problem, saveat=1)
    # Extract the 2nd state, the (I)infected, for the dates with observations.
    sol_for_observed = sol[2, 2:15]
    # Compute the sum-of-squares of the infected vs. data.
    return sum(abs2.(sol_for_observed - data.in_bed))
end
```

    loss (generic function with 1 method)

And the go-to for optimization in Julia is [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/)

```julia
using Optim
# An alternative to writing `y -> f(x, y)` is `Base.Fix1(f, x)` which
# avoids potential performance issues with global variables (as our `problem` here).
opt = optimize(
    p -> loss(problem, p), # function to minimize
    [0, 0],                # lower bounds on variables
    [Inf, Inf],            # upper bounds on variables
    [2.0, 0.5],            # initial values
    Fminbox(NelderMead())  # optimization alg
) 
```

    * Status: success
    
    * Candidate solution
       Final objective value:     4.116433e+03
    
    * Found with
       Algorithm:     Fminbox with Nelder-Mead
    
    * Convergence measures
       |x - x'|               = 0.00e+00 ≤ 0.0e+00
       |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
       |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
       |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
       |g(x)|                 = 7.86e+04 ≰ 1.0e-08
    
    * Work counters
       Seconds run:   4  (vs limit Inf)
       Iterations:    4
       f(x) calls:    565
       ∇f(x) calls:   1

```julia
β, λ = Optim.minimizer(opt)
```

    2-element Vector{Float64}:
     1.6692320164955483
     0.44348639177622445

```julia
# Solve for the obtained parameters.
problem = remake(problem, p=(β, λ))
sol = solve(problem)

# Plot the solution.
plot(sol, linewidth=2, xaxis="Time in days", label=["Suspectible" "Infected" "Recovered"], alpha=0.5)
# And the data.
scatter!(1:14, data.in_bed, label="Data", color="black")
```

![img](./.ob-jupyter/d16ee5c3ab7072e15e7efe28a8f390104175913c.png)

That's better than our *totally* "random" guess from earlier!


## Example: SEIR model

Now we're going to add another compartment to our SIR model: the *(E)xposed* state

\begin{equation*}
\begin{split}
  \dv{S}{t} &= - \beta S \frac{I}{N} \\
  \dv{E}{t} &= \frac{\beta I S}{N} - \sigma E \\
  \dv{I}{t} &= \sigma E - \gamma I \\
  \dv{R}{t} &= \gamma I
\end{split}
\end{equation*}

where we've added a new parameter \(\sigma\) describing the fraction of people who develop observable symptoms in this time


## TASK Solve the SEIR model using Julia

```julia
function SEIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    N = 763  # population

    S, E, I, R = u  # have ourselves an additional state!
    β, γ, σ = p     # and an additional parameter!

    # TODO: Implement yah fool!
    du[1] = nothing
    du[2] = nothing
    du[3] = nothing
    du[4] = nothing
end
```

**BONUS:** Use `Optim.jl` to find minimizers of sum-of-squares


## SOLUTION Solve the SEIR model using Julia

```julia
function SEIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t    # current time
)
    N = 763  # population
    S, E, I, R = u  # have ourselves an additional state!
    β, γ, σ = p     # and an additional parameter!

    # Might as well cache these computations.
    βSI = β * S * I / N
    σE = σ * E
    γI = γ * I

    du[1] = -βSI
    du[2] = βSI - σE
    du[3] = σE - γI
    du[4] = γI
end
```

    SEIR! (generic function with 1 method)

```julia
problem_seir = let u0 = [N - 1, 0, 1, 0], β = 2.0, γ = 0.6, σ = 0.8
    ODEProblem(SEIR!, u0, tspan, (β, γ, σ))
end
```

    ODEProblem with uType Vector{Int64} and tType Float64. In-place: true
    timespan: (0.0, 14.0)
    u0: 4-element Vector{Int64}:
     762
       0
       1
       0

```julia
sol_seir = solve(problem_seir, saveat=1)
```

    retcode: Success
    Interpolation: 1st order linear
    t: 15-element Vector{Float64}:
      0.0
      1.0
      2.0
      3.0
      4.0
      5.0
      6.0
      7.0
      8.0
      9.0
     10.0
     11.0
     12.0
     13.0
     14.0
    u: 15-element Vector{Vector{Float64}}:
     [762.0, 0.0, 1.0, 0.0]
     [760.1497035901518, 1.277915971753478, 1.0158871356490553, 0.5564933024456415]
     [757.5476928906271, 2.425869618233348, 1.6850698824327135, 1.341367608706787]
     [753.081189706403, 4.277014534677882, 2.9468385687120784, 2.6949571902067637]
     [745.3234082630842, 7.455598293492679, 5.155811621098981, 5.065181822323938]
     [731.9851682751213, 12.855816151849933, 8.960337047554939, 9.198678525473571]
     [709.5042941973462, 21.77178343781762, 15.384985521594787, 16.338936843241182]
     [672.8733895183619, 35.77263271085456, 25.88133104438007, 28.472646726403138]
     [616.390571176038, 55.97177756967422, 42.09614416178476, 48.54150709250279]
     [536.453596476594, 81.2428045994271, 64.9673325777641, 80.33626634621449]
     [436.43708330634297, 106.04037246704702, 92.9550757379631, 127.56746848864664]
     [329.60092931771436, 121.08020372279418, 120.48402926084937, 191.83483769864185]
     [233.8471941518982, 119.43669383157659, 139.3233304893263, 270.3927815271987]
     [160.88805352426687, 102.7399386960996, 143.3826208089892, 355.9893869706441]
     [111.72261866282292, 79.02493776169311, 132.78384886713565, 439.46859470834806]

```julia
plot(sol_seir, linewidth=2, xaxis="Time in days", label=["Suspectible" "Exposed" "Infected" "Recovered"], alpha=0.5)
scatter!(1:14, data.in_bed, label="Data")
```

![img](./.ob-jupyter/9c5ac26d982f01e7094445522f39583d90db8c3f.png)

Don't look so good. Let's try Optim.jl again.

```julia
function loss_seir(problem, p)
    problem = remake(problem, p=p)
    sol = solve(problem, saveat=1)
    # NOTE: 3rd state is now the (I)nfectious compartment!!!
    sol_for_observed = sol[3, 2:15]
    return sum(abs2.(sol_for_observed - data.in_bed))
end
```

    loss_seir (generic function with 1 method)

```julia
opt = optimize(Base.Fix1(loss_seir, problem_seir), [0, 0, 0], [Inf, Inf, Inf], [2.0, 0.5, 0.9], Fminbox(NelderMead()))
```

    * Status: success (reached maximum number of iterations)
    
    * Candidate solution
       Final objective value:     3.115978e+03
    
    * Found with
       Algorithm:     Fminbox with Nelder-Mead
    
    * Convergence measures
       |x - x'|               = 0.00e+00 ≤ 0.0e+00
       |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
       |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
       |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
       |g(x)|                 = 1.77e+05 ≰ 1.0e-08
    
    * Work counters
       Seconds run:   2  (vs limit Inf)
       Iterations:    3
       f(x) calls:    13259
       ∇f(x) calls:   1

```julia
β, γ, σ = Optim.minimizer(opt)
```

    3-element Vector{Float64}:
     4.853872993924619
     0.4671485850111774
     0.8150294098438762

```julia
sol_seir = solve(remake(problem_seir, p=(β, γ, σ)), saveat=1)
plot(sol_seir, linewidth=2, xaxis="Time in days", label=["Suspectible" "Exposed" "Infected" "Recovered"], alpha=0.5)
scatter!(1:14, data.in_bed, label="Data", color="black")
```

![img](./.ob-jupyter/76e3d91d3de473fb85975cd9a6e88513b634e79d.png)

> But&#x2026;but these are <span class="underline">point estimates</span>! What about distributions? WHAT ABOUT UNCERTAINTY?!

No, no that's fair.

Let's do some Bayesian inference then.

BUT FIRST!


## Making our future selves less annoyed

It's a bit annoying to have to have all these different functions for both `SIR!` and `SEIR!`

<div class="fragment (appear)">

```julia
# Abstract type which we can use to dispatch on.
abstract type AbstractEpidemicProblem end

struct SIRProblem{P} <: AbstractEpidemicProblem
    problem::P
    N::Int
end

function SIRProblem(N::Int; u0 = [N - 1, 1, 0.], tspan = (0, 14), p = [2.0, 0.6])
    return SIRProblem(ODEProblem(SIR!, u0, tspan, p), N)
end
```

    SIRProblem

Then we can just construct the problem as

```julia
sir = SIRProblem(N);
```

</div>

And to make it a bit easier to work with, we add some utility functions

```julia
# General.
parameters(prob::AbstractEpidemicProblem) = prob.problem.p
initial_state(prob::AbstractEpidemicProblem) = prob.problem.u0
population(prob::AbstractEpidemicProblem) = prob.N

# Specializations.
susceptible(::SIRProblem, u::AbstractMatrix) = u[1, :]
infected(::SIRProblem, u::AbstractMatrix) = u[2, :]
recovered(::SIRProblem, u::AbstractMatrix) = u[3, :]
```

    recovered (generic function with 1 method)

So that once we've solved the problem, we can easily extract the compartment we want, e.g.

```julia
sol = solve(sir.problem, saveat=1)
infected(sir, sol)
```

    15-element Vector{Float64}:
       1.0
       4.026799533924021
      15.824575905720002
      56.779007685250534
     154.4310579906169
     248.98982384839158
     243.67838619968524
     181.93939659551987
     120.64627375763271
      75.92085282572398
      46.58644927641269
      28.214678599716418
      16.96318676577873
      10.158687874394722
       6.070415830241046


## TASK Implement `SEIRProblem`

```julia
struct SEIRProblem <: AbstractEpidemicProblem
    # ...
end

function SEIRProblem end

susceptible
exposed
infected
recovered
```


## SOLUTION Implement `SEIRProblem`

```julia
struct SEIRProblem{P} <: AbstractEpidemicProblem
    problem::P
    N::Int
end

function SEIRProblem(N::Int; u0 = [N - 1, 0, 1, 0.], tspan = (0, 14), p = [4.5, 0.45, 0.8])
    return SEIRProblem(ODEProblem(SEIR!, u0, tspan, p), N)
end

susceptible(::SEIRProblem, u::AbstractMatrix) = u[1, :]
exposed(::SEIRProblem, u::AbstractMatrix) = u[2, :]
infected(::SEIRProblem, u::AbstractMatrix) = u[3, :]
recovered(::SEIRProblem, u::AbstractMatrix) = u[4, :]
```

    recovered (generic function with 2 methods)

Now, given a `problem` and a `sol`, we can query the `sol` for the `infected` state without explicit handling of which `problem` we're working with

```julia
seir = SEIRProblem(N);
sol = solve(seir.problem, saveat=1)
infected(seir, sol)
```

    15-element Vector{Float64}:
       1.0
       1.9941817088874336
       6.958582307202902
      23.9262335176065
      74.23638542794971
     176.98368495653585
     276.06126059898344
     293.92632518571605
     249.92836195453708
     189.07578975511504
     134.2373192679034
      91.82578430804273
      61.38108478932363
      40.42264366743211
      26.357816296754425


# Bayesian inference

First off

```julia
using Turing
```

This dataset really doesn't have too many observations

```julia
nrow(data)
```

    14

So reporting a single number for parameters is maybe being a *bit* too confident

We'll use the following model

\begin{equation*}
\begin{split}
  \beta &\sim \mathcal{N}_{ + }(2, 1) \\
  \gamma &\sim \mathcal{N}_{ + }(0.4, 0.5) \\
  \phi^{-1} &\sim \mathrm{Exponential}(1/5) \\
   y_i &\sim \mathrm{NegativeBinomial2}\big(F(u_0, t_i;\ \beta, \gamma), \phi \big)
\end{split}
\end{equation*}

where

-   \(\big( y_i \big)_{i = 1}^{14}\) are the observations,
-   \(F\) is the integrated system, and
-   \(\phi\) is the over-dispersion parameter.

```julia
plot(
    plot(truncated(Normal(2, 1); lower=0), label=nothing, title="β"),
    plot(truncated(Normal(0.4, 0.5); lower=0), label=nothing, title="γ"),
    plot(Exponential(1/5), label=nothing, title="ϕ⁻¹"),
    layout=(3, 1)
)
```

![img](./.ob-jupyter/c7a2ffd1b9f12c5932b50fe0c4b7923253c2d1bc.png)

```julia
function NegativeBinomial2(mean, ϕ)
    p = 1/(1 + mean/ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end
```

    NegativeBinomial2 (generic function with 1 method)

```julia
# Let's just make sure we didn't do something stupid.
μ = 2; ϕ = 3;
dist = NegativeBinomial2(μ, ϕ)
# Source: https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
mean(dist) ≈ μ && var(dist) ≈ μ + μ^2 / ϕ
```

    true

Can be considered a generalization of `Poisson`

```julia
μ = 2.0
anim = @animate for ϕ ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
    p = plot(size=(500, 300))
    plot!(p, Poisson(μ); label="Poisson($μ)")
    plot!(p, NegativeBinomial2(μ, ϕ), label="NegativeBinomial2($μ, $ϕ)")
    xlims!(0, 20); ylims!(0, 0.35);
    p
end
gif(anim, "negative_binomial.gif", fps=2);
```

    [ Info: Saved animation to /home/tor/Projects/public/Turing-Workshop/2023-Geilo-Winter-School/03-advanced-turing/negative_binomial.gif

![img](./.ob-jupyter/75415f5b0c05009b61a3b811092fac87856c171f.png)

![img](./negative_binomial.gif)

```julia
@model function sir_model(
    num_days;                                  # Number of days to model
    tspan = (0.0, float(num_days)),            # Timespan to model
    u0 = [N - 1, 1, 0.0],                      # Initial state
    p0 = [2.0, 0.6],                           # Placeholder parameters
    problem = ODEProblem(SIR!, u0, tspan, p0)  # Create problem once so we can `remake`.
)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)
    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem, p=[β, γ])  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)       # Solve!

    sol_for_observed = sol[2, 2:num_days + 1]  # Timesteps we have observations for.
    in_bed = Vector{Int}(undef, num_days)
    for i = 1:length(sol_for_observed)
        # Add a small constant to `sol_for_observed` to make things more stable.
        in_bed[i] ~ NegativeBinomial2(sol_for_observed[i] + 1e-5, ϕ)
    end

    # Some quantities we might be interested in.
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
end
```

    sir_model (generic function with 2 methods)

Let's break it down

```julia
β ~ truncated(Normal(2, 1); lower=0)
γ ~ truncated(Normal(0.4, 0.5); lower=0)
ϕ⁻¹ ~ Exponential(1/5)
ϕ = inv(ϕ⁻¹)
```

defines our prior

`truncated` is just a way of restricting the domain of the distribution you pass it

```julia
problem_new = remake(problem, p=[β, γ])  # Replace parameters `p`.
sol = solve(problem_new, saveat=1)       # Solve!
```

We then remake the problem, now with the parameters `[β, γ]` sampled above

Remember the `saveat = 1` ensures we get solution at timesteps `[0, 1, 2, ..., 14]`

Then we extract the timesteps we have observations for

```julia
sol_for_observed = sol[2, 2:num_days + 1]  # Timesteps we have observations for.
```

and define what's going to be a likelihood (once we add observations)

```julia
in_bed = Vector{Int}(undef, num_days)
for i = 1:length(sol_for_observed)
    # Add a small constant to `sol_for_observed` to make things more stable.
    in_bed[i] ~ NegativeBinomial2(sol_for_observed[i] + 1e-5, ϕ)
end
```

Finally we return some values that might be of interest to

```julia
# Some quantities we might be interested in.
return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
```

This is useful for a post-sampling diagnostics, debugging, etc.

```julia
model = sir_model(length(data.in_bed))
```

    Model(
      args = (:num_days, :tspan, :u0, :p0, :problem)
      defaults = (:tspan, :u0, :p0, :problem)
      context = DynamicPPL.DefaultContext()
    )

The model is just another function, so we can call it to check that it works

<div class="fragment (appear)">

```julia
model().infected
```

    14-element Vector{Float64}:
       3.1664025930805666
       9.798841386086416
      28.31455092148007
      68.31877539019936
     115.4174366684663
     124.74895879419394
      95.20892597910733
      59.60522553801775
      33.75389615433325
      18.17904383289882
       9.563348320777385
       4.96497417899723
       2.560551848705429
       1.317201355058522

Hey, it does!

</div>


## Is the prior reasonable?

Before we do any inference, we should check if the prior is reasonable

From domain knowledge we know that (for influenza at least)

-   \(R_0\) is typically between 1 and 2
-   `recovery_time` (\(1 / \gamma\)) is usually ~1 week

<div class="fragment (appear)">

We want to make sure that your prior belief reflects this knowledge while still being flexible enough to accommodate the observations

</div>

To check this we'll just simulate some draws from our prior model, i.e. the model *without* conditioning on `in_bed`

There are two ways to sample form the prior

<div class="fragment (appear)">

```julia
# 1. By just calling the `model`, which returns a `NamedTuple` containing the quantities of interest
print(model())
```

    (R0 = 1.2999290756779531, recovery_time = 0.6977054268642389, infected = [1.5292966105765635, 2.3223774580679386, 3.4895335689371967, 5.161081785767661, 7.459321870754651, 10.433526681897439, 13.956211547134568, 17.620261444577014, 20.74293231144959, 22.578921692450297, 22.667291359227267, 21.068926446896786, 18.29922357837848, 15.019852319846871])

</div>

<div class="fragment (appear)">

Or by just calling `sample` using `Prior`

```julia
# Sample from prior.
chain_prior = sample(model, Prior(), 10_000);
```

    Sampling: 100%|█████████████████████████████████████████| Time: 0:00:01

</div>

```julia
using StatsPlots

"""
    plot_trajectories!(p::Plots.Plot, chain::MCMCChains.Chains; kwargs...)

Plot trajectories in `chain`.

# Keyword arguents
n = 1000, trajectory_color="#BBBBBB", alpha=1/√n
- `n`: number of trajectories to sample. Default: 1000.
- `trajectory_color`: color of the trajectories to use. Default: "#BBBBBB".
- `alpha`: alpha to use when plotting the trajectories. Default: `1/√n`.
- `include_data`: include the data in the plot. Default: `false`.
- `tspan`: 2-tuple containing first and last time represented by `chain`. Default: (1, 14)
"""
function plot_trajectories!(
    p::Plots.Plot, chain::MCMCChains.Chains;
    n = 1000, include_data=false, tspan=(1, 14),
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

    if include_data
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
```

```julia
"""
    plot_trajectory_quantiles!(p::Plots.Plot, chain::MCMCChains.Chains; kwargs...)

Plot quantiles of trajectories in `chain`.

By default, the 95% quantiles are used.

# Keyword arguments
- q: quantiles to compute (only three ordered values supported). Default: `[0.025, 0.5, 0.975]`.
- `include_data`: if `true`, include the data in the plot. Default: `false`.
- `tspan`: 2-tuple containing first and last time represented by `chain`. Default: (1, 14)
"""
function plot_trajectory_quantiles!(
    p::Plots.Plot, chain::MCMCChains.Chains;
    q=[0.025, 0.5, 0.975], include_data=false, tspan=(1, 14)
)
    # Compute quantiles.
    qs = Array(DataFrame(quantile(chain; q = q))[:, Not(:parameters)]);

    # Plot!
    num_timesteps = size(qs, 1)
    lb, mid, ub = qs[:, 1], qs[:, 2], qs[:, 3]
    plot!(p, tspan[1]:tspan[2], mid, ribbon=(mid - lb, ub - mid))

    if include_data
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
```

Let's have a look at the prior predictive

```julia
p = plot(legend=false, size=(600, 300))
plot_trajectories!(p, group(chain_prior, :in_bed); n = 1000)
hline!([N], color="red")
```

![img](./.ob-jupyter/898d8baf1113c3a719d020702e40e75c9c545b22.png)

For certain values we get number of infected *larger* than the actual population

But this is includes the randomness from `NegativeBinomial2` likelihood

Maybe more useful to inspect the (I)nfected state from the ODE solution?

We can also look at the `generated_quantities`, i.e. the values from the `return` statement in our model

```julia
quantities_prior = generated_quantities(model, MCMCChains.get_sections(chain_prior, :parameters))
print(quantities_prior[1])
```

    (R0 = 31.06593882253981, recovery_time = 9.924227728868157, infected = [20.135496891976175, 267.5387933367594, 638.3259546937711, 632.4208659251071, 575.6411969096877, 520.778868925039, 470.89294272632065, 425.7604715323202, 384.95061249728093, 348.05199150979433, 314.6901297730605, 284.52608963917, 257.25336296664375, 232.59481447030583])

This is simply a `Matrix` of `NamedTuple`

```julia
typeof(quantities_prior)
```

    Matrix{NamedTuple{(:R0, :recovery_time, :infected), Tuple{Float64, Float64, Vector{Float64}}}} (alias for Array{NamedTuple{(:R0, :recovery_time, :infected), Tuple{Float64, Float64, Array{Float64, 1}}}, 2})

We can convert it into a `Chains` using a utility function of mine

```julia
chain_quantities_prior = to_chains(quantities_prior);

p = plot(legend=false, size=(600, 300))
plot_trajectories!(p, group(chain_quantities_prior, :infected); n = 1000)
hline!([N], color="red")
```

![img](./.ob-jupyter/5bc3a173351c09a67a80b68fdc2f3b9267ef7af4.png)

<div class="x-small-text">

**NOTE:** `to_chains` is not part of "official" Turing.jl because the `return` can contain *whatever* you want, and so it's not always possible to convert into a `Chains`

</div>

And the quantiles for the trajectories

```julia
p = plot(legend=false, size=(600, 300))
plot_trajectory_quantiles!(p, group(chain_quantities_prior, :infected))
hline!(p, [N], color="red")
```

![img](./.ob-jupyter/21acfe1e8d2d2784548d455e81fbae739072fd28.png)

```julia
DataFrame(quantile(chain_quantities_prior[:, [:R0, :recovery_time], :]))
```

<div><div style = "float: left;"><span>2×6 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">parameters</th><th style = "text-align: left;">2.5%</th><th style = "text-align: left;">25.0%</th><th style = "text-align: left;">50.0%</th><th style = "text-align: left;">75.0%</th><th style = "text-align: left;">97.5%</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">R0</td><td style = "text-align: right;">0.514661</td><td style = "text-align: right;">2.1126</td><td style = "text-align: right;">3.69669</td><td style = "text-align: right;">7.2244</td><td style = "text-align: right;">59.5376</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">recovery_time</td><td style = "text-align: right;">0.705466</td><td style = "text-align: right;">1.19966</td><td style = "text-align: right;">1.8608</td><td style = "text-align: right;">3.44473</td><td style = "text-align: right;">29.4863</td></tr></tbody></table></div>

Compare to our prior knowledge of \(R_0 \in [1, 2]\) and \((1/\gamma) \approx 1\) for influenza

Do we really need probability mass on \(R_0 \ge 10\)?


## TASK What's wrong with the current prior?

Here's the SIR model

\begin{equation*}
\begin{split}
  \dv{S}{t} &= - \beta S \frac{I}{N} \\
  \dv{I}{t} &= \beta S \frac{I}{N} - \gamma I \\
  \dv{R}{t} &= \gamma I
\end{split}
\end{equation*}


## SOLUTION Recovery time shouldn't be several years

We mentioned recovery time, which is expressed as \(1 / \gamma\), is ~1 week

While we're clearly putting non-zero probability on regions near 0, i.e. *long* recovery times

```julia
plot(truncated(Normal(0.4, 0.5); lower=0), label=nothing, title="γ", size=(500, 300))
```

![img](./.ob-jupyter/0c811fb684f86d4d0f68b2e7a7d1836e293f54e2.png)

We should probably be putting less probability mass near 0


## SOLUTION \(\gamma\) should not be larger than 1

\begin{equation*}
\begin{split}
  \dv{S}{t} &= - \beta S \frac{I}{N} \\
  \dv{I}{t} &= \beta S \frac{I}{N} - {\color{red} \gamma I} \\
  \dv{R}{t} &= {\color{red} \gamma I}
\end{split}
\end{equation*}

If \(\gamma > 1\) ⟹ (R)ecovered increase by *more* than the (I)nfected

⟹ <span class="underline">healthy people are recovering</span>

Now, I'm no epidemiologist, but that doesn't seem right

Maybe something like

```julia
plot(Beta(2, 5), label="new", size=(500, 300))
plot!(truncated(Normal(0.4, 0.5); lower=0), label="old", color="red")
```

![img](./.ob-jupyter/fa0872fb9b3d3beea9866c1760af900ffcadef47.png)

-   [X] Bounded at 1
-   [X] Allows smaller values (i.e. longer recovery time) but rapidly decreases near zero


## SOLUTION What if \(\beta > N\)?

Then for \(t = 0\) we have

\begin{equation*}
\dv{S}{t} \bigg|_{t = 0} = - \beta S \frac{I}{N} > - N (N - 1) \frac{1}{N} = - (N - 1)
\end{equation*}

i.e. we *immediately* infect everyone on the very first time-step

Also doesn't seem very realistic

*But* under our current prior does this matter?

```julia
# ℙ(β > N) = 1 - ℙ(β ≤ N)
1 - cdf(truncated(Normal(2, 1); lower=0), N)
```

    0.0

Better yet

```julia
quantile(truncated(Normal(2, 1); lower=0), 0.95)
```

    3.6559843567138275

i.e. 95% of the probability mass falls below ~3.65

⟹ Current prior for \(\beta\) is fine (✓)

Before we change the prior, let's also make it a bit easier to change the prior using `@submodel`

<div class="fragment (appear)">

`@submodel` allows you call models within models, e.g.

```julia
@model function A()
    x_hidden_from_B ~ Normal()
    x = x_hidden_from_B + 100
    return x
end

@model function B()
    @submodel x = A()
    y ~ Normal(x, 1)

    return (; x, y)
end
```

    B (generic function with 2 methods)

</div>

<div class="fragment (appear)">

```julia
# So if we call `B` we only see `x` and `y`
println(B()())
```

    (x = 100.6700504957998, y = 100.59882836474439)

</div>

<div class="fragment (appear)">

```julia
# While if we sample from `B` we get the latent variables
println(rand(B()))
```

    (x_hidden_from_B = 1.1620740097515572, y = 100.82791610035471)

</div>

And if you want to make sure you avoid clashes of variable-names, we can specify a `prefix`

```julia
@model A() = (x ~ Normal(); return x + 100)

@model function B()
    # Given it a prefix to use for the variables in `A`.
    @submodel prefix=:inner x_inner = A()
    x ~ Normal(x_inner, 1)

    return (; x_inner, x)
end
```

    B (generic function with 2 methods)

```julia
print(rand(B()))
```

    (var"inner.x" = 0.5588685738879569, x = 102.56600691107558)

`@submodel` is useful as it allows you to:

1.  Easy to swap out certain parts of your model.
2.  Can re-use models across projects and packages.

When working on larger projects, this really shines

Equipped with `@submodel` we can replace

```julia
β ~ truncated(Normal(2, 1); lower=0)
γ ~ truncated(Normal(0.4, 0.5); lower=0)
```

with

```julia
@submodel p = prior(problem_wrapper)
```

<div class="fragment (appear)">

where `prior` can be something like

```julia
@model function prior_original(problem_wrapper::SIRProblem)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)

    return [β, γ]
end

@model function prior_improved(problem_wrapper::SIRProblem)
    # NOTE: Should probably also lower mean for `β` since
    # more probability mass on small `γ` ⟹ `R0 =  β / γ` grows.
    β ~ truncated(Normal(1, 1); lower=0)
    # NOTE: New prior for `γ`.
    γ ~ Beta(2, 5)

    return [β, γ]
end
```

    prior_improved (generic function with 2 methods)

</div>

```julia
@model function epidemic_model(
    problem_wrapper::AbstractEpidemicProblem,
    prior  # NOTE: now we just pass the prior as an argument
)
    # NOTE: And use `@submodel` to embed the `prior` in our model.
    @submodel p = prior(problem_wrapper)

    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem_wrapper.problem, p=p)  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)                  # Solve!

    # Extract the `infected`.
    sol_for_observed = infected(problem_wrapper, sol)[2:end]

    # NOTE: `arraydist` is faster for larger dimensional problems,
    # and it does not require explicit allocation of the vector.
    in_bed ~ arraydist(NegativeBinomial2.(sol_for_observed .+ 1e-5, ϕ))

    β, γ = p[1:2]
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed, in_bed = in_bed)
end
```

<div class="x-small-text">

Another neat trick is to return early if integration fail

</div>

```julia
@model function epidemic_model(
    problem_wrapper::AbstractEpidemicProblem,
    prior  # now we just pass the prior as an argument
)
    # And use `@submodel` to embed the `prior` in our model.
    @submodel p = prior(problem_wrapper)

    ϕ⁻¹ ~ Exponential(1/5)
    ϕ = inv(ϕ⁻¹)

    problem_new = remake(problem_wrapper.problem, p=p)  # Replace parameters `p`.
    sol = solve(problem_new, saveat=1)                  # Solve!

    # NOTE: Return early if integration failed.
    if !issuccess(sol)
        Turing.@addlogprob! -Inf  # NOTE: Causes automatic rejection.
        return nothing
    end

    # Extract the `infected`.
    sol_for_observed = infected(problem_wrapper, sol)[2:end]

    # `arraydist` is faster for larger dimensional problems,
    # and it does not require explicit allocation of the vector.
    in_bed ~ arraydist(NegativeBinomial2.(sol_for_observed .+ 1e-5, ϕ))

    β, γ = p[1:2]
    return (R0 = β / γ, recovery_time = 1 / γ, infected = sol_for_observed)
end
```

    epidemic_model (generic function with 2 methods)

Equipped with this we can now easily construct *two* models using different priors

```julia
sir = SIRProblem(N);
model_original = epidemic_model(sir, prior_original);
model_improved = epidemic_model(sir, prior_improved);
```

but using the same underlying `epidemic_model`

```julia
chain_prior_original = sample(model_original, Prior(), 10_000; progress=false);
chain_prior_improved = sample(model_improved, Prior(), 10_000; progress=false);
```

Let's compare the resulting priors over some of the quantities of interest

Let's compare the `generated_quantities`, e.g. \(R_0\)

<div class="small-text">

```julia
chain_quantities_original = to_chains(
    generated_quantities(
        model_original,
        MCMCChains.get_sections(chain_prior_original, :parameters)
    );
);

chain_quantities_improved = to_chains(
    generated_quantities(
        model_improved,
        MCMCChains.get_sections(chain_prior_improved, :parameters)
    );
);
```

</div>

```julia
p = plot(; legend=false, size=(500, 200))
plot_trajectories!(p, group(chain_quantities_original, :infected); n = 100, trajectory_color="red")
plot_trajectories!(p, group(chain_quantities_improved, :infected); n = 100, trajectory_color="blue")
hline!([N], color="red", linestyle=:dash)
```

![img](./.ob-jupyter/26821687da0892468280513773b9a64d7dce9405.png)

```julia
plt1 = plot(legend=false)
plot_trajectory_quantiles!(plt1, group(chain_quantities_original, :infected))
hline!(plt1, [N], color="red", linestyle=:dash)

plt2 = plot(legend=false)
plot_trajectory_quantiles!(plt2, group(chain_quantities_improved, :infected))
hline!(plt2, [N], color="red", linestyle=:dash)

plot(plt1, plt2, layout=(2, 1))
```

![img](./.ob-jupyter/aeead7124fd9708aec4e02205c817050c831e511.png)

This makes sense: if half of the population is immediately infected ⟹ number of infected tapers wrt. time as they recover

For `model_improved` we then have

```julia
DataFrame(quantile(chain_quantities_improved[:, [:R0, :recovery_time], :]))
```

<div><div style = "float: left;"><span>2×6 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">parameters</th><th style = "text-align: left;">2.5%</th><th style = "text-align: left;">25.0%</th><th style = "text-align: left;">50.0%</th><th style = "text-align: left;">75.0%</th><th style = "text-align: left;">97.5%</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">R0</td><td style = "text-align: right;">0.280317</td><td style = "text-align: right;">2.20343</td><td style = "text-align: right;">4.42919</td><td style = "text-align: right;">8.41909</td><td style = "text-align: right;">34.8588</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">recovery_time</td><td style = "text-align: right;">1.55993</td><td style = "text-align: right;">2.52591</td><td style = "text-align: right;">3.75755</td><td style = "text-align: right;">6.12444</td><td style = "text-align: right;">22.7184</td></tr></tbody></table></div>

Compare to `model_original`

```julia
DataFrame(quantile(chain_quantities_original[:, [:R0, :recovery_time], :]))
```

<div><div style = "float: left;"><span>2×6 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">parameters</th><th style = "text-align: left;">2.5%</th><th style = "text-align: left;">25.0%</th><th style = "text-align: left;">50.0%</th><th style = "text-align: left;">75.0%</th><th style = "text-align: left;">97.5%</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">R0</td><td style = "text-align: right;">0.526962</td><td style = "text-align: right;">2.14448</td><td style = "text-align: right;">3.75959</td><td style = "text-align: right;">7.23873</td><td style = "text-align: right;">58.1674</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">recovery_time</td><td style = "text-align: right;">0.70578</td><td style = "text-align: right;">1.22111</td><td style = "text-align: right;">1.88519</td><td style = "text-align: right;">3.51881</td><td style = "text-align: right;">29.2338</td></tr></tbody></table></div>


## TASK Make `epidemic_model` work for `SEIRProblem`

1.  [ ] Implement a prior which also includes \(\sigma\) and execute `epidemic_model` with it
2.  [ ] Can we make a better prior for \(\sigma\)? Do we even need one?

```julia
@model function prior_original(problem_wrapper::SEIRProblem)
    # TODO: Implement
end
```

```julia
model_seir = epidemic_model(SEIRProblem(N), prior_original)
model_seir()
```


## SOLUTION 

```julia
@model function prior_original(problem_wrapper::SEIRProblem)
    β ~ truncated(Normal(2, 1); lower=0)
    γ ~ truncated(Normal(0.4, 0.5); lower=0)
    σ ~ truncated(Normal(0.8, 0.5); lower=0)

    return [β, γ, σ]
end
```

    prior_original (generic function with 4 methods)

```julia
model_seir = epidemic_model(SEIRProblem(N), prior_original)
print(model_seir())
```

    (R = 1.0808660851871192, recovery_time = 0.7841248305093967, in_bed = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], infected = [0.4999668242687421, 0.46235766285215496, 0.47507248907963295, 0.49368945024837424, 0.5134221032371306, 0.5337402038407291, 0.554583530003258, 0.5759276332283051, 0.5977185406043448, 0.6200612777280481, 0.6426263512877511, 0.666016344720367, 0.6893325151132493, 0.7135486410591751])


## WARNING Consult with domain experts

<span class="underline">This guy should not be the one setting your priors!</span>

![img](.notes/attachments/A_litle_Norwegian_boy/2023-01-18_14-49-24_471337_3317365246956_1262712540_o.jpg)

Get an actual scientist to do that&#x2026;


## Condition

Now let's actually involve the data

<div class="fragment (appear)">

We can condition a `Model` as so

```julia
# Condition on the observations.
model = epidemic_model(SIRProblem(N), prior_improved)
model_conditioned = model | (in_bed = data.in_bed,)
```

    Model(
      args = (:problem_wrapper, :prior)
      defaults = ()
      context = ConditionContext(
        (in_bed = [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4],),
        DynamicPPL.DefaultContext()
      )
    )

</div>

<div class="fragment (appear)">

You know what time it is: *inference time*!

</div>


## Metropolis-Hastings (MH)

```julia
chain_mh = sample(model_conditioned, MH(), MCMCThreads(), 10_000, 4; discard_initial=5_000);
```

```julia
chain_mh
```

    Chains MCMC chain (10000×4×4 Array{Float64, 3}):
    
    Iterations        = 5001:1:15000
    Number of chains  = 4
    Samples per chain = 10000
    Wall duration     = 1.13 seconds
    Compute duration  = 4.5 seconds
    parameters        = β, γ, ϕ⁻¹
    internals         = lp
    
    Summary Statistics
      parameters      mean       std   naive_se      mcse        ess      rhat   e ⋯
          Symbol   Float64   Float64    Float64   Float64    Float64   Float64     ⋯
    
               β    1.7190    0.0514     0.0003    0.0029   176.0675    1.0402     ⋯
               γ    0.5270    0.0390     0.0002    0.0022   168.3655    1.0443     ⋯
             ϕ⁻¹    0.1395    0.0716     0.0004    0.0039   185.0000    1.0101     ⋯
                                                                    1 column omitted
    
    Quantiles
      parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
          Symbol   Float64   Float64   Float64   Float64   Float64 
    
               β    1.6296    1.6792    1.7128    1.7454    1.8310
               γ    0.4433    0.5046    0.5275    0.5542    0.6001
             ϕ⁻¹    0.0414    0.0889    0.1325    0.1841    0.3112

Rhat is *okay-ish* but not great, and ESS is pretty low innit?

```julia
plot(chain_mh; size=(800, 500))
```

![img](./.ob-jupyter/d944644d54b9d9a76d97087f1f99a907567e3ba2.png)

Eeehh doesn't look the greatest

Difficult to trust these results, but let's check if it at least did *something* useful

```julia
# We're using the unconditioned model!
predictions_mh = predict(model, chain_mh)
```

    Chains MCMC chain (10000×14×4 Array{Float64, 3}):
    
    Iterations        = 1:1:10000
    Number of chains  = 4
    Samples per chain = 10000
    parameters        = in_bed[1], in_bed[2], in_bed[3], in_bed[4], in_bed[5], in_bed[6], in_bed[7], in_bed[8], in_bed[9], in_bed[10], in_bed[11], in_bed[12], in_bed[13], in_bed[14]
    internals         = 
    
    Summary Statistics
      parameters       mean       std   naive_se      mcse          ess      rhat 
          Symbol    Float64   Float64    Float64   Float64      Float64   Float64 
    
       in_bed[1]     3.2663    2.2024     0.0110    0.0154   33111.3354    1.0004
       in_bed[2]    10.6208    5.3730     0.0269    0.0779    3133.5610    1.0036
       in_bed[3]    33.1050   15.4850     0.0774    0.3424    1192.1830    1.0071
       in_bed[4]    90.6461   41.7400     0.2087    1.0924     814.9446    1.0119
       in_bed[5]   183.6523   78.8518     0.3943    1.7730    1128.0863    1.0097
       in_bed[6]   246.9214   97.7439     0.4887    1.5320    2697.8115    1.0039
       in_bed[7]   237.3656   91.2831     0.4564    1.0442    6332.8471    1.0008
       in_bed[8]   187.7543   73.6327     0.3682    0.8712    5331.2879    1.0010
       in_bed[9]   134.3697   53.6737     0.2684    0.7836    3171.1837    1.0014
      in_bed[10]    91.3608   37.3504     0.1868    0.6280    2126.5619    1.0022
      in_bed[11]    60.5226   25.6094     0.1280    0.4792    1676.7385    1.0034
      in_bed[12]    39.7219   17.6551     0.0883    0.3769    1250.0428    1.0047
      in_bed[13]    25.8260   12.1288     0.0606    0.2662    1208.1916    1.0039
      in_bed[14]    16.5964    8.3344     0.0417    0.1887    1111.3682    1.0050
    
    Quantiles
      parameters      2.5%      25.0%      50.0%      75.0%      97.5% 
          Symbol   Float64    Float64    Float64    Float64    Float64 
    
       in_bed[1]    0.0000     2.0000     3.0000     5.0000     8.0000
       in_bed[2]    2.0000     7.0000    10.0000    14.0000    23.0000
       in_bed[3]   10.0000    23.0000    31.0000    41.0000    69.0000
       in_bed[4]   31.0000    63.0000    84.0000   110.0000   190.0000
       in_bed[5]   67.0000   131.0000   172.0000   222.0000   370.0000
       in_bed[6]   93.0000   180.0000   235.0000   298.0000   475.0000
       in_bed[7]   90.0000   175.0000   227.0000   287.0000   446.0000
       in_bed[8]   69.0000   138.0000   179.0000   227.0000   359.0000
       in_bed[9]   49.0000    98.0000   128.0000   163.0000   258.0000
      in_bed[10]   32.0000    66.0000    87.0000   111.0000   178.0000
      in_bed[11]   21.0000    43.0000    57.0000    74.0000   120.0000
      in_bed[12]   13.0000    28.0000    37.0000    49.0000    81.0000
      in_bed[13]    7.0000    17.0000    24.0000    32.0000    54.0000
      in_bed[14]    4.0000    11.0000    15.0000    21.0000    36.0000

```julia
plot_trajectories!(plot(legend=false), predictions_mh; include_data=true)
```

![img](./.ob-jupyter/bbd3face0b041d4acc92f00d8a9c45db1d96c643.png)

```julia
plot_trajectory_quantiles!(plot(legend=false, size=(600, 300)), predictions_mh; include_data=true)
```

![img](./.ob-jupyter/cca448d55db7f5ea0cfc74ddbcaf1a71b6bb8aee.png)

Okay, it's not *completely* useless, but my trust-issues are still present.

Metropolis-Hastings have disappointed me one too many times before.

Before we proceed, let's just make some functions for the visualizations


## So instead, let's go `NUTS`

That's right, we're reaching to the **No U-Turn sampler (NUTS)**


### 

<https://chi-feng.github.io/mcmc-demo/app.html>


## 

> Wooaah there! `NUTS` requires <span class="underline">gradient information</span>!
> 
> How are you going to get that through that `solve`?

Good question, voice in my head

I'm obviously not going to it myself


## Automatic differentiation (AD) in Julia

-   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): forward-mode AD *(default in Turing.jl)*
-   [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl): tape-based reverse-mode AD
-   [Zygote.jl](https://github.com/FluxML/Zygote.jl): source-to-source reverse-mode AD
-   And more&#x2026;

Up-and-coming

-   [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl): Julia bindings for [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) which ADs LLVM (low-level)
-   [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl): experimental mixed-mode AD meant to replace Zygote.jl

Of importance

-   [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl): light-weight package for defining rules, compatible with many of the above

**Important**

> When you write code, you don't have to make a choice which one you want to use!

All the (stable) ones, will (mostly) work

*But* how you write code will affect performance characteristics

Takes a bit of know-how + a bit of digging to go properly "vroom!"


## Differentiating through `solve`

With that being said, differentiating through numerical `solve` is not necessarily trivial to do efficiently

There are numerous ways of approaching this problem

![img](.notes/attachments/Bayesian_inference/2023-01-22_12-30-07_Screenshot_20230122_122936.png)

<https://arxiv.org/abs/1812.01892> is *great* resource

<div class="fragment (appear)">

But this is why we have [`SciMLSensitivity.jl`](https://github.com/SciML/SciMLSensitivity.jl)

[SciMLSensitivity.jl docs](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#Choosing-a-Sensitivity-Algorithm) also provides a great overview of different approaches

</div>

```julia
using SciMLSensitivity
```

let's you make use of the `sensealg` keyword in `solve` to choose whatever approach you need

It offers

1.  *Discrete sensitivity analysis* or the *"Direct" method*: just use `ForwardDiff.Dual` in the `solve`.
2.  *Continuous local sensitivity analysis (CSA)*: extends the original system such that the `solve` gives you both the solution and the the gradient simultaenously.
3.  *Adjoint methods*: construct a backwards system whose solution gives us the gradient.
    -   Here you can use anything from symbolically derived to "AD-derived" jacobians.


## Back to being `NUTS`

```julia
chain = sample(model_conditioned, NUTS(0.8), MCMCThreads(), 1000, 4);
```

    ┌ Info: Found initial step size
    └   ϵ = 0.05
    ┌ Info: Found initial step size
    └   ϵ = 0.05
    ┌ Info: Found initial step size
    └   ϵ = 0.05
    ┌ Warning: Instability detected. Aborting
    └ @ SciMLBase ~/.julia/packages/SciMLBase/QqtZA/src/integrator_interface.jl:525
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, false, true)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    ┌ Warning: Instability detected. Aborting
    └ @ SciMLBase ~/.julia/packages/SciMLBase/QqtZA/src/integrator_interface.jl:525
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, false, true)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    ┌ Info: Found initial step size
    └   ϵ = 0.0125
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    Sampling (4 threads): 100%|█████████████████████████████| Time: 0:00:01

```julia
chain
```

    Chains MCMC chain (1000×15×4 Array{Float64, 3}):
    
    Iterations        = 501:1:1500
    Number of chains  = 4
    Samples per chain = 1000
    Wall duration     = 36.99 seconds
    Compute duration  = 143.14 seconds
    parameters        = β, γ, ϕ⁻¹
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size
    
    Summary Statistics
      parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
          Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯
    
               β    1.7308    0.0544     0.0009    0.0011   2372.6172    0.9993    ⋯
               γ    0.5291    0.0442     0.0007    0.0008   2922.1364    0.9999    ⋯
             ϕ⁻¹    0.1364    0.0753     0.0012    0.0017   2097.7887    1.0009    ⋯
                                                                    1 column omitted
    
    Quantiles
      parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
          Symbol   Float64   Float64   Float64   Float64   Float64 
    
               β    1.6251    1.6974    1.7288    1.7630    1.8475
               γ    0.4410    0.5009    0.5279    0.5571    0.6144
             ϕ⁻¹    0.0425    0.0837    0.1195    0.1706    0.3392

Muuuch better! Both ESS and Rhatc is looking good

```julia
plot(chain; size=(800, 500))
```

![img](./.ob-jupyter/f198b9606c06a59ed28e9000e05a436b045fda5c.png)

```julia
# Predict using the results from NUTS.
predictions = predict(model, chain)
```

    Chains MCMC chain (1000×14×4 Array{Float64, 3}):
    
    Iterations        = 1:1:1000
    Number of chains  = 4
    Samples per chain = 1000
    parameters        = in_bed[1], in_bed[2], in_bed[3], in_bed[4], in_bed[5], in_bed[6], in_bed[7], in_bed[8], in_bed[9], in_bed[10], in_bed[11], in_bed[12], in_bed[13], in_bed[14]
    internals         = 
    
    Summary Statistics
      parameters       mean       std   naive_se      mcse         ess      rhat 
          Symbol    Float64   Float64    Float64   Float64     Float64   Float64 
    
       in_bed[1]     3.3685    2.2097     0.0349    0.0313   4135.6883    1.0008
       in_bed[2]    10.8680    5.4725     0.0865    0.1047   3548.2111    0.9997
       in_bed[3]    34.1945   15.8405     0.2505    0.2548   3773.1286    1.0003
       in_bed[4]    92.7943   42.9267     0.6787    0.6561   3451.8320    1.0008
       in_bed[5]   187.4745   76.9431     1.2166    0.9506   3723.5739    0.9994
       in_bed[6]   248.7530   98.1284     1.5515    1.7181   3902.0437    1.0003
       in_bed[7]   235.6747   90.1544     1.4255    1.4267   3907.9089    1.0002
       in_bed[8]   185.3945   70.5461     1.1154    1.0862   3765.5992    0.9997
       in_bed[9]   131.7980   52.3720     0.8281    0.8008   3826.4520    0.9993
      in_bed[10]    88.0415   36.5904     0.5785    0.5883   3853.5288    0.9994
      in_bed[11]    59.4323   25.8624     0.4089    0.3847   3983.9082    0.9997
      in_bed[12]    38.2265   17.4133     0.2753    0.2263   3627.0453    0.9996
      in_bed[13]    24.9842   12.3378     0.1951    0.2193   3397.6433    0.9998
      in_bed[14]    16.2740    8.7979     0.1391    0.1472   3270.0093    1.0001
    
    Quantiles
      parameters      2.5%      25.0%      50.0%      75.0%      97.5% 
          Symbol   Float64    Float64    Float64    Float64    Float64 
    
       in_bed[1]    0.0000     2.0000     3.0000     5.0000     9.0000
       in_bed[2]    3.0000     7.0000    10.0000    14.0000    23.0000
       in_bed[3]   11.0000    23.0000    32.0000    42.0000    72.0000
       in_bed[4]   31.0000    64.0000    87.0000   112.0000   198.0250
       in_bed[5]   68.0000   136.0000   176.0000   227.0000   366.0000
       in_bed[6]   92.0000   182.0000   238.0000   300.0000   474.0250
       in_bed[7]   92.0000   175.0000   225.0000   282.0000   447.0000
       in_bed[8]   71.0000   135.0000   177.0000   226.2500   345.0250
       in_bed[9]   48.9750    95.7500   124.0000   161.0000   255.0000
      in_bed[10]   32.0000    63.0000    83.0000   107.0000   174.0250
      in_bed[11]   21.0000    41.0000    56.0000    72.0000   118.0250
      in_bed[12]   11.0000    26.0000    36.0000    47.0000    78.0000
      in_bed[13]    7.0000    17.0000    23.0000    31.0000    53.0000
      in_bed[14]    4.0000    10.0000    15.0000    20.0000    37.0000

```julia
plot_trajectories!(plot(legend=false, size=(600, 300)), predictions; n = 1000, include_data=true)
```

![img](./.ob-jupyter/5818a2844c7cdbb2eaa4ad8c8cf1e0756273a161.png)

```julia
plot_trajectory_quantiles!(plot(legend=false, size=(600, 300)), predictions; include_data=true)
```

![img](./.ob-jupyter/813c664e623143776099024e0e736e4fe49800d5.png)


## Simulation-based calibration (SBC) [Talts et. al. (2018)](https://arxiv.org/abs/1804.06788)

1.  Sample from prior \(\theta_1, \dots, \theta_n \sim p(\theta)\).
2.  Sample datasets \(\mathcal{D}_i \sim p(\cdot \mid \theta_i)\) for \(i = 1, \dots, n\).
3.  Obtain (approximate) \(p(\theta \mid \mathcal{D}_i)\) for \(i = 1, \dots, n\).

For large enough \(n\), the "combination" of the posteriors should recover the prior!

"Combination" here usually means computing some statistic and comparing against what it should be

![img](.notes/attachments/Bayesian_inference/2023-01-22_12-09-24_Screenshot_20230122_120848.png)

That's very expensive → in practice we just do this once or twice

```julia
# Sample from the conditioned model so we don't get the `in_bed` variables too
using Random  # Just making usre the numbers of somewhat interesting
rng = MersenneTwister(43);
test_values = rand(rng, NamedTuple, model_conditioned)
```

|    |    |                   |    |    |                    |    |    |                    |
|--- |--- |------------------ |--- |--- |------------------- |--- |--- |------------------- |
| β | = | 1.2254566808077714 | γ | = | 0.27594266205681933 | ϕ⁻¹ | = | 0.13984179162984164 |

Now we condition on those values and run once to generate data

```julia
model_test = model | test_values
```

    Model(
      args = (:problem_wrapper, :prior)
      defaults = ()
      context = ConditionContext(
        (β = 1.2254566808077714, γ = 0.27594266205681933, ϕ⁻¹ = 0.13984179162984164),
        DynamicPPL.DefaultContext()
      )
    )

```julia
in_best_test = rand(rng, model_test).in_bed;
```

Next, inference!

```julia
model_test_conditioned = model | (in_bed = in_best_test,)
```

    Model(
      args = (:problem_wrapper, :prior)
      defaults = ()
      context = ConditionContext(
        (in_bed = [1, 9, 11, 45, 159, 136, 270, 123, 463, 376, 231, 148, 99, 162],),
        DynamicPPL.DefaultContext()
      )
    )

```julia
# Let's just do a single chain here.
chain_test = sample(model_test_conditioned, NUTS(0.8), 1000);
```

    ┌ Info: Found initial step size
    └   ϵ = 0.8
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/4fByY/src/hamiltonian.jl:49
    Sampling: 100%|█████████████████████████████████████████| Time: 0:00:01

Did we recover the parameters?

```julia
ps = []
for sym in [:β, :γ, :ϕ⁻¹]
    p = density(chain_test[:, [sym], :])
    vline!([test_values[sym]])
    push!(ps, p)
end
plot(ps..., layout=(3, 1), size=(600, 400))
```

![img](./.ob-jupyter/c7f5262cc2962e2920df901926b9172e5bbfb589.png)


## TODO Choosing AD backend in Turing.jl


# TODO Showing off


## `Gibbs`

Simple linear regression model

```julia
using LinearAlgebra: I

@model function linear_regression(X)
    num_params = size(X, 1)
    β ~ MvNormal(ones(num_params))
    σ² ~ InverseGamma(2, 3)
    y ~ MvNormal(vec(β' * X), σ² * I)
end

# Generate some dummy data.
X = randn(2, 10_000)
lin_reg = linear_regression(X)
true_vals = rand(lin_reg)

# Condition.
lin_reg_conditioned = lin_reg | (y = true_vals.y,)
```

    Model(
      args = (:X,)
      defaults = ()
      context = ConditionContext(
        (y = [0.3242032989648078, 1.0927828698345088, -0.7332477681370562, -5.062789065066403, -0.5510236918192273, 3.849292275620334, -3.1453844688196386, -1.0761023645067103, -0.35920155367909423, 1.1632762258546507  …  1.557561970476609, 3.575676520486435, 0.07694907753939595, -2.102027398588007, -8.469490043524004, 3.4605534596978993, 4.7196306430050665, -1.7976651698542838, -0.6009882061621101, 1.397842282973244],),
        DynamicPPL.DefaultContext()
      )
    )

Can of course just use `HMC` on it

```julia
chain_hmc = sample(lin_reg_conditioned, HMC(1e-3, 32), 1_000);
MCMCChains.ess_rhat(chain_hmc)
```

    Sampling: 100%|█████████████████████████████████████████| Time: 0:00:02
    ESS
      parameters        ess      rhat   ess_per_sec 
          Symbol    Float64   Float64       Float64 
    
            β[1]   433.7061    1.0007      155.9533
            β[2]   434.2894    1.0014      156.1630
              σ²   749.5058    1.0003      269.5095

But, how about we combine **Elliptical Slice Sampling (`ESS`)**, which is very effective on Gaussian priors, together *with* `HMC` on \(\sigma^2\) (which is non-Gaussian and so ESS doesn't apply)

```julia
chain_ess_hmc = sample(lin_reg_conditioned, Gibbs(ESS(:β), HMC(1e-3, 32, :σ²)), 1_000);
MCMCChains.ess_rhat(chain_ess_hmc)
```

    Sampling: 100%|█████████████████████████████████████████| Time: 0:00:02
    ESS
      parameters        ess      rhat   ess_per_sec 
          Symbol    Float64   Float64       Float64 
    
            β[1]    68.1919    1.0140       32.6433
            β[2]   170.8266    1.0017       81.7744
              σ²   156.3485    1.0043       74.8437

This is much improved (depending on which random seed you're using)!


# Concluding remarks


## TODO Julia: The Good, the Bad, and the Ugly

An honest take from a little Norwegian boy


### The Good

-   Speed
-   Composability (thank you multiple dispatch)
-   No need to tie yourself to an underlying computational framework (looking at you Python)
-   Interactive
-   *Transparency*

Most of these you have already, but the transparency is maybe not something we've seen too much of

```julia
f(x) = 2x
```

    f (generic function with 1 method)

```julia
f(1)
```

    2

You can inspect the type-inferred and lowered code

```julia
@code_typed f(1)
```

    CodeInfo(
    1 ─ %1 = Base.mul_int(2, x)::Int64
    └──      return %1
    ) => Int64

You can inspect the LLVM code

```julia
@code_llvm f(1)
```

    ;  @ In[2]:1 within `f`
    define i64 @julia_f_2659(i64 signext %0) #0 {
    top:
    ; ┌ @ int.jl:88 within `*`
       %1 = shl i64 %0, 1
    ; └
      ret i64 %1
    }

And even the resulting machine code

```julia
@code_native f(1)
```

    	.text
    	.file	"f"
    	.globl	julia_f_2696                    # -- Begin function julia_f_2696
    	.p2align	4, 0x90
    	.type	julia_f_2696,@function
    julia_f_2696:                           # @julia_f_2696
    ; ┌ @ In[2]:1 within `f`
    	.cfi_startproc
    # %bb.0:                                # %top
    ; │┌ @ int.jl:88 within `*`
    	leaq	(%rdi,%rdi), %rax
    ; │└
    	retq
    .Lfunc_end0:
    	.size	julia_f_2696, .Lfunc_end0-julia_f_2696
    	.cfi_endproc
    ; └
                                            # -- End function
    	.section	".note.GNU-stack","",@progbits

It really just depends on which level of "I hate my life" you're currently at


### The Bad


### The Ugly


### Overall


# References

-   <https://mc-stan.org/users/documentation/case-studies/boarding_school_case_study.html>
