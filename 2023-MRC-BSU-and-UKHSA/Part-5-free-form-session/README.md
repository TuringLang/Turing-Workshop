# Free-form session

In this session, we set up a "research" project from scratch and started doing a bit of inference.

## A research project from scratch

Here's the guide we followed to set up the project `MyResearch`.

### Setting up the project environment

(1) Navigate to directory where you want the project to be, for example `/tmp`.

```sh
# Or whatever directory you want to be in
cd /tmp
```

(2) `]` to get into Pkg

You should see something like
```julia
(@v1.9) pkg> 
```

(3) From here, you type `generate MyResearch`

That is,

```julia
(@v1.9) pkg> generate MyResearch
```

The following should be your output

```
  Generating  project MyResearch:
    MyResearch/Project.toml
    MyResearch/src/MyResearch.jl
```

(4) Navigate into `MyResearch/`

You can do this by hitting `;` to get into the shell mode, which should look like this

```julia
shell> 
```

and typing `path/to/MyResearch`, e.g.

```julia
shell> cd MyResearch/
```

(5) Now that our working project is the project directory , we can activate the project environment.

Again, enter Pkg mode using `]`. You should see

```julia
pkg> 
```

And from here you can type `activate .`, where the `.` indicates the current folder. That is,

```julia
pkg> activate .
```

You should now see

```julia
  Activating project at `/tmp/MyResearch`
```

(6) If you now type `status`, i.e.

```julia
pkg> status
```

you should see an empty environment, e.g.

```julia
Project MyResearch v0.1.0
Status `/tmp/MyResearch/Project.toml` (empty project)
```

(7) From here we can add dependenices that we want, e.g. type `add Turing DataFrames StatsPlots DifferentialEquations Distributions Dates LinearAlgebra CSV`

```julia
(MyResearch) pkg> add Turing DataFrames StatsPlots DifferentialEquations Distributions Dates LinearAlgebra CSV
```

You subsequently see the following output

```julia
   Resolving package versions...
    Updating `/tmp/adsfas/MyResearch/Project.toml`
  [336ed68f] + CSV v0.10.11
  [a93c6f00] + DataFrames v1.6.1
  [0c46a032] + DifferentialEquations v7.9.1
  [31c24e10] + Distributions v0.25.100
  [f3b207a7] + StatsPlots v0.15.6
  [fce5fe82] + Turing v0.29.1
  [ade2ca70] + Dates
  [37e2e46d] + LinearAlgebra
    Updating `/tmp/adsfas/MyResearch/Manifest.toml`
...
```

(8) If you run `status` you should now see something like

```julia
(MyResearch) pkg> status
Project MyResearch v0.1.0
Status `/tmp/MyResearch/Project.toml`
  [336ed68f] CSV v0.10.11
  [a93c6f00] DataFrames v1.6.1
  [0c46a032] DifferentialEquations v7.9.1
  [31c24e10] Distributions v0.25.100
  [f3b207a7] StatsPlots v0.15.6
  [fce5fe82] Turing v0.29.1
  [ade2ca70] Dates
  [37e2e46d] LinearAlgebra
```

### Editing code

If we look at the files in the project, we'll see the following:

```sh
├── Manifest.toml
├── Project.toml
└── src
    └── MyResearch.jl

1 directory, 3 files
```

We can see that in `src/` there is already a file called `MyResearch.jl`.

Let's see what's already in it:

```julia
module MyResearch

greet() = print("Hello World!")

end # module MyResearch
```

We can actually access this from our Julia session / REPL now!

```julia
julia> using MyResearch

julia> MyResearch.greet()
Hello World!
```

If we then edit `src/MyResearch.jl` to, say,

```julia
module MyResearch

greet() = print("Hello MRC!")

end # module MyResearch
```

and go back to the session / REPL, we can re-run our `greet` function:

```julia
julia> MyResearch.greet()
Hello MRC!
```

## Live-coding

The result of the live-coding you can find in:
- `MyResearch/src/MyResearch.jl`: module `MyResearch` + `load_data`.
- `MyResearch/scripts/inference.jl`: script where we did the following:
  - Perform inference on for the SIR model with influenza data from the presentation.
  - Condition the model only one the first 7 days, perform inference, and then use some of initial states from this to perform inference on the subsequent 7 days.
