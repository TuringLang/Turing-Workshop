# Getting started

We'll be using Julia **1.9.3** throughout, so we recommend you use the same version to avoid any issues. 
(If the patch version is different, e.g. 1.9.2 or v1.9.1, that is no problem; the important bit is that you're on a version ≥1.9)

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

For example, on Linux and mac this is generally achieved by running the command `julia` or the executable `/wherever/you/installed/julia/bin/julia` at the terminal.

# Editor Choice

You are free to use your preferred editor (emacs, vims, sublime, notepad++ etc), however, we generally recommend either
1. [Visual Studio (VS) Code](https://code.visualstudio.com/) with [the Julia extensions](https://www.julia-vscode.org/), or
2. [Jupyter notebooks](https://jupyter.org/), which is easily supported by simply installing the Julia package `IJulia`; once this is installed you should find a `julia` kernel available in your notebook.[^1]

The latter option might be more familiar if you're coming from a scientific Python background, but VS Code is generally the recommended option.
Amongst us speakers we're familiar with Jupyter notebooks, Will and Xianda are VS Code users, and Tor is a (grumpy) `emacs` user, hence we should be able to provide _some_ support with a few different setups.

# Additional resources
- Julia Language website: https://www.julialang.org
- Turing website: https://turing.ml/stable/
- Julia Language YouTube channel: https://www.youtube.com/thejulialanguage

[^1]: Alternatively, one can start the notebook from the Julia REPL by running `using IJulia; IJulia.notebook()`.
