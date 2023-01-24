# 2023 Geilo Winter School - Turing Workshop

Geilo Winter School, Geilo Norway, 2023-01-22 to 2023-01-27.

## Instructors

- [Jose Storopoli](https://github.com/storopoli)
- [Tor Erlend Fjelde](https://github.com/torfjelde)

## Contents

1. Julia/Turing Basics (90 mins)

   - Quick intro to Julia
     - Multiple Dispatch: Motivating example
   - Basics of Turing
     - Overview of ecosystem and packages

1. Using Turing in practice (90 mins)

   - How to define Bayesian/generative models
   - Different MCMC samplers
   - What to do after sampling (post-processing)
   - Diagnostics
   - Predictive Checks

1. Advanced Turing usage (90 mins)

   - Performance of Turing models
   - Debugging Turing of models
   - ODE in Turing models

1. Cutting-edge MCMC algorithms research with Turing (90 mins)

   - How to use the exposed interfaces in the Turing ecossystem
   - Case example of Parallel MCMC Tempering

## Getting set up with Julia

See [setting-up-julia.md](./setting-up-julia.md) for how to get set up with Julia.

## Datasets

- `cheese` (hierarchical models): data from cheese ratings.
   A group of 10 rural and 10 urban raters rated 4 types of different cheeses (A, B, C and D) in two samples.
   Source: Boatwright, P., McCulloch, R., & Rossi, P. (1999). Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model. _Journal of the American Statistical Association_, 94(448), 1063–1073.
