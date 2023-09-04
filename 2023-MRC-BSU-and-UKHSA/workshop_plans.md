# Plans for MRC-BSU and UKHSA Workshop

## Timetable (Un-synced with [Google Doc](https://docs.google.com/document/d/1B7dVCFCMMugH13rRfyjzUo7vCfBBwWwi0D_NsNmfuX0/edit))

### Thursday 21st - General Introduction

 Time       | Activity                                   |
------------|--------------------------------------------|
 10.00      | Introduction                               |
 10.15-11.15| Introduction to Julia                      |
 11.15-11.30| Break                                      |
 11.30-12.30| Introduction to Julia                      |
 12.30-13.30| Lunch                                      |
 13.30-14.30| Introduction to Turing                     |
 14.30-15.00| Break                                      |
 15.00-16.00| Introduction to Turing                     |
 16.00-17.15| One and half hour practical/discussion/hands on |

### Friday 22nd - Case Studies

 Time       | Activity                                   |
------------|--------------------------------------------|
 10.00-10.15| Summary from previous day                   |
 10.15-11.15| JuliaBUGS                                  |
 11.15-11.30| Break                                      |
 11.30-12.30| Turing/JuliaBUGS + SciML (?)               |
 12.30-13.30| Lunch                                      |
 13.30-15.00| Gaussian processes                         |
 15.00-15.45| Break                                      |
 15.45-17.15| Turing/JuliaBUGS + SciML (?)               |

## Topics

### Julia Intro

> General goal is to get people to the point where they can read and understand Julia code, and write simple Julia code.

#### Basics (1 hour)

- Brief recap Julia installation and development environment (10min)
  - Julia REPL
  - VSCode
    - Julia Extension
      - Debugger
      - Formatter
    - Other Extensions
      - Jupyter Notebook in VSCode
      - Copilot
- Essential Julia Syntax: bare minimum of Julia syntax to understand Julia code (30min)
  - Basic data-type
  - Controls: if, for, while
  - Functions
  - Data structure
    - Arrays
    - Tuples and NamedTuples
    - Dictionary
- Simple Julia "Case Study" (20min)
  - Give some examples and walk through the code

#### (Slightly) Advanced (1 hour)

- Macros
- Multiple dispatch and types
- A gentle intro to the Julia ecosystem for various things, including statistics, probability distributions, differential equations, etc.
- Plotting
  - Mainly Plots.jl and its descendants
  - Mention AlgebraOfPlotting.jl and Makie.jl?
- DataFrames.jl
- Distributions.jl
  - Mention SpecialFunctions.jl, StatsFuns.jl, etc.
- SciML ecosystem
- Automatic Differentiation
- Julia in practice (in no particular order)
  - Benchmarking / Profiling
  - Chasing type-stability
  - Widely adopted conventions, e.g. `!` and `!!`
  - Setting up a project

### Turing.jl (2 hours)

### Gaussian Processes (1.5 hours)

### JuliaBUGS.jl (1 - 1.5 hour)

- Start with case study
  - Center on one example
  - Goal
    - Not a conceptual introduction to the package, but maybe on probabilistic modeling with Directed Probabilistic Model; JuliaBUS is targeted for this kind of usage
  - Assume Tor already introduced packages in Turing ecosystem and Julia
  - From graphical model (on the paper) to BUGS program
  - Model compilations and inference APIs
  - Visualization
    - **TODO**: implement plotting plate notation
  - Work with MetaGraph
    - Implementing algorithm based on graphical model, for instance particle MCMC
  - Graph-based inference: state-space model, e.g. HMM?
- Introduce BUGS syntax
  - Not too much, can get boring
- The second example should be SIR-related, but maybe we should combine this case study with the Turing/JuliaBUGS-SciML integration session

### Turing/JuliaBUGS-SciML

- [Tor’s material](https://github.com/TuringLang/Turing-Workshop/blob/main/2023-Geilo-Winter-School/Part-2-Turing-and-other-things/notes.ipynb) is really good
- Some possible revisions
  - **TODO**: If the audiences are experts in epidemiology already, how should we adjust focus?
- Maybe mention but not focus on JuliaBUGS by show a BUGS model
- Do we mention Individual level models? (e.g. Agent.jl, Pathogen.jl)
