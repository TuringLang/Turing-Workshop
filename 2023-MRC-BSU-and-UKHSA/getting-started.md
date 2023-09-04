# Getting started

These workshops are intended to be somewhat interactive, and so you might be at risk of having to actually write and execute some code on your own device!

If you're already familiar with Julia and have Julia 1.8 or greater already installed, then you can probably stop reading here.

If you're new to Julia, you will unfortunately have to install yet another language on your device. Luckily, it's a fairly straight forward mission.

## "Woah there! I have to *execute* Julia code? HOW?!" - Installing Julia

We'll be using Julia **1.9.x** throughout, so we recommend you use the same version to avoid any issues.

If you're using a Windows machine, we highly recommend checking out the [`Windows Subsystem for Linux (WSL)`](https://learn.microsoft.com/en-us/windows/wsl/install). It's a powerful tool that allows you to run Linux applications, including GUI apps, and even supports GPU acceleration. This can greatly enhance your Julia programming experience.

**Julia can be downloaded at the official website (https://julialang.org/downloads/) and a more detailed instructions can also be found there (https://julialang.org/downloads/platform/).**

*Other than the official installers, you can also try [`Juliaup`](https://github.com/JuliaLang/juliaup) which will make it easier to work with multiple Julia versions down the line.*

At the end of this process you should have access to some way of getting you into the Julia REPL which should present something similar to this:

```julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.3 (2023-08-24)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 
```

For example, on a Linux device this is generally achieved by running the command `julia` or the executable `/wherever/you/installed/julia/bin/julia`.

(If the version is slightly different, e.g. 1.9.2, that is no problem; the important bit is that you're on a version ≥1.9)

## "But how do I *write* Julia code?!" - Setting up an editor

As mentioned earlier, at certain points during these workshops you're likely to find yourself in a position where you even need to *write* some Julia code. I know, this is all very demanding.

How and where and with what you want to write these chunks of text is of course completely up to yourself.

- If you live and die by Notepad++, then you're welcome to use Notepad++.
- If you're one of those wizards who doesn't have to spend 30 minutes trying to exit a file you opened in `vim`, then maybe you should use `vim`.
- Or if you're part of the opposition of the group of people described in the previous line, i.e. an `emacs` user, then you might want to use `emacs`.

**But most (sane) people will probably prefer something like:**
1. a proper IDE, in which case [Visual Studio (VS) Code](https://code.visualstudio.com/) with [the Julia extensions](https://www.julia-vscode.org/) is a great option, or
2. [Pluto.jl](https://github.com/fonsp/Pluto.jl), which is a reactive notebook specifically for and implemented in Julia. This has the benefit of being trivial to set up + the underlying file of the notebook is a simple Julia file, so it's possible to execute it both as a script and view it as a nice notebook in the browser.
3. [Jupyter notebooks](https://jupyter.org/), which is easily supported by simply installing the Julia package `IJulia`; once this is installed you should find a `julia` kernel available in your notebook.[^1]

The latter option might be more familiar if you're coming from a scientific Python background, but VS Code is generally the recommended option.

Amongst us speakers we're familiar with Jupyter notebooks, Will and Xianda are VS Code users, and Tor is a (grumpy) `emacs` user, hence we should be able to provide _some_ support with a few different setups.

Note that we expect everyone to act civil during the workshops, regardless of the differences in choices of editor people make. 
In particular, any criticism of `emacs` will be met by loud boos and shaking of fists from Tor in an attempt to quell potential conflicts.

# Additional resources
- Julia Language website: https://www.julialang.org
- Turing website: https://turing.ml/stable/
- Julia Language YouTube channel: https://www.youtube.com/thejulialanguage

[^1]: Alternatively, one can start the notebook from the Julia REPL by running `using IJulia; IJulia.notebook()`.
