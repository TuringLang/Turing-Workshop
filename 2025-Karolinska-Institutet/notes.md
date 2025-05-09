## Talk Structure and Pedagogy

start with the goal and background
State the intended learning goals of the talk
start with a conceptual introduction: Julia, PPL, Bayesian workflow
What are you assuming about the audience’ knowledge?
You mentioned likelihood without defining it
But then you did define it, but very fast

## Explanations of Key Concepts

Explain why taking logs is useful?
S(t) = Pr(T>t)
h(t) = - d/dt log S(t)
Maybe show these on the same plot - or state intuition for the hazard h(t)
What is Kaplan-Meier - comes out of the blue!
If NCCTG is pretty famous, give a date or a reference. Useful context.

## Model Implementation and Turing-Specifics

type of `y` should be generic
Re the @model syntax, perhaps show the two interpretations of the model.
Sampling is one interpretation. Random variables are another.
The subroutine with individual_frailty was a bit confusing - more to be explained.
Maybe just inline Gamma(k,k) rather than do the function call.
The numeric goals on the results of MCMC seem quite arbitrary - any intuition? Maybe not!

## Comparisons with Other PPLs and Tools

JuliaBUGS in the beginning, then move to Turing
And JuliaBUGS - why not?
maybe show stan model to communicate the model and later compare stan and turing
BridgeStan -- stan in julia
eager eval of Turing compared to Stan
Maybe mention Infer.NET

## Visualization Suggestions

How about a plot of count of survivors first,
And then show the plot of probabilities

## Resources and References

references:
colab
Include some URLs to find out more.
