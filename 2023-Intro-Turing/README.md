# 2023 Bayesian Data Analysis Meetup Group - Introduction to Turing

Bayesian Data Analysis Meetup Group, Online, 2023-02-23.

Link: <https://www.meetup.com/bda-group/events/290842675/>

## Instructors

- [Jose Storopoli](https://github.com/storopoli)

## Contents

1. Define a Model --- Basic PPL (Probabilistic Programming Language) syntax
   1. Deterministics and probabilistic assignments
   1. avoiding `for`-loops and efficient coding
1. Fit a model
   1. MCMC sampler choice
   1. Multi-threaded and Multiprocess
1. Inspecting chains --- built-in functionality
1. Visualizing chains --- built-in functionality

## Getting set up with Julia

See [setting_up_julia.md](../setting_up_julia.md) for how to get set up with Julia.

## Datasets

- `wells` (logistic regression): Data from a survey of 3,200 residents in a small
  area of Bangladesh suffering from arsenic contamination of groundwater.
  Respondents with elevated arsenic levels in their wells had been encouraged
  to switch their water source to a safe public or private well in the nearby
  area and the survey was conducted several years later to learn which of the
  affected residents had switched wells.
  Source: Gelman, A., & Hill, J. (2007). Data analysis using regression and
  multilevel/hierarchical models. Cambridge university press.
