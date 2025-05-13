## Data

using DataFrames, RDatasets
using Statistics: mean, std
using Plots.PlotMeasures

# Load original data
df_lung = dataset("survival", "cancer")

# Summary of the data
DataFrames.describe(df_lung)

# Define columns of interest
vars_to_check = [:Time, :Status, :Age, :Sex]

# Remove rows with missing values in columns of interest
df = dropmissing(df_lung, vars_to_check)

# Standardize Age
standardise(x) = (x .- mean(x)) ./ std(x)
age_std = standardise(df.Age)

# Create boolean indicator for Female (Sex == 2)
female = df.Sex .== 2

# Create design matrix X with intercept, standardized Age, and Female indicator
X = hcat(ones(nrow(df)), age_std, female)

# Extract outcome variable: survival time
y = Float64.(df.Time)

# Create event indicator (true if event/death occurred, false if censored)
# Convention: Status=1 -> censored, Status=2 -> dead (event)
event = df.Status .== 2

## Kaplan-Meier

using StatsPlots # Ensure plotting library is loaded
using Plots # Ensure Plots is loaded for yflip functionality
using Survival # Ensure Survival is loaded

# Fit overall Kaplan-Meier estimator using the original (unsorted) full data
km_overall_fit = fit(KaplanMeier, y, event)

# Sort the data by time for visualization purposes
sort_indices = sortperm(y)
y_sorted = y[sort_indices]
event_sorted = event[sort_indices]
# Create indices for the y-axis representing patients sorted by time
indices_sorted = 1:length(y)

# Plot 3: Visualize the full data (sorted by time) and the overall KM curve
# Create the scatter plot for the full data
p_full_km = scatter(
    y_sorted[.!event_sorted],
    indices_sorted[.!event_sorted],
    marker = :circle,
    label = "Censored",
    color = :blue,
    markersize = 4,
    alpha = 0.7,
    legend = :topright,
)
scatter!(
    p_full_km,
    y_sorted[event_sorted],
    indices_sorted[event_sorted],
    marker = :cross,
    label = "Event (Death)",
    color = :red,
    markersize = 5,
    alpha = 0.7,
)
plot!(
    p_full_km,
    xlabel = "Observed Time (days)",
    ylabel = "Patient Index (Sorted by Time)",
    title = "Patient Survival Time and Overall Kaplan-Meier Estimate",
    yflip = true, # Flip y-axis to show time progression downwards
)

# Add the overall KM curve on a secondary y-axis
p_twin_full = twinx(p_full_km)
plot!(
    p_twin_full,
    km_overall_fit.events.time,
    km_overall_fit.survival,
    seriestype = :steppost,
    label = "Kaplan-Meier (Overall)",
    color = :black,
    linewidth = 2,
    ylabel = "Survival Probability",
    legend = :bottomleft, # Position legend for KM curve
    ylims = (0, 1.05), # Ensure y-axis for probability is 0 to 1,
    # Add margin to the right side of the plot to accommodate the secondary y-axis labels
    left_margin = 20mm,
    right_margin = 30mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    size = (800, 600),  # Set an appropriate size for the plot
)

# Display the final plot for the full data
p_full_km

##

p_data = scatter(
    y_sorted[.!event_sorted],
    indices_sorted[.!event_sorted],
    marker = :circle,
    label = "Censored",
    color = :blue,
    markersize = 4,
    alpha = 0.7,
    legend = :topright,
)
scatter!(
    p_data,
    y_sorted[event_sorted],
    indices_sorted[event_sorted],
    marker = :cross,
    label = "Event (Death)",
    color = :red,
    markersize = 5,
    alpha = 0.7,
)
plot!(
    p_data,
    xlabel = "Observed Time (days)",
    ylabel = "Patient Index (Sorted by Time)",
    title = "Patient Survival Time",
    yflip = true, # Flip y-axis to show time progression downwards
    size = (800, 600),
    left_margin = 20mm,
    right_margin = 30mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
)

##

function plot_weibull_curves(
    param_sets;
    t_max = 10.0,
    n_points = 200,
    plot_size = (1600, 800),
)
    """
    Generates and displays plots for Weibull survival S(t) and hazard h(t) functions
    for different sets of parameters.

    Args:
        param_sets: A list or array of tuples. Each tuple should contain:
                    (alpha, theta, label_str)
                    alpha (Float64): Shape parameter of the Weibull distribution.
                    theta (Float64): Scale parameter of the Weibull distribution.
                    label_str (String): A label for this parameter set in the plot legend.
                                        If label_str is empty or not provided, a default label
                                        based on alpha and theta values will be generated.
        t_max (Float64): Maximum time value for the x-axis. Default is 10.0.
        n_points (Int): Number of points to calculate along the time axis. Default is 200.
        plot_size (Tuple{Int, Int}): Size of the combined plot. Default is (800,800).

    Returns:
        A combined plot object from Plots.jl with two subplots:
        1. Survival functions S(t) for each parameter set.
        2. Hazard functions h(t) for each parameter set.
    """

    # Define Weibull survival function S(t)
    # S(t) = exp(-(t/theta)^alpha)
    function weibull_survival(t, alpha, theta)
        if t < 0.0
            return 1.0 # Survival is 1 before t=0
        end
        # alpha and theta are assumed > 0 from checks in the main loop
        return exp(-(t / theta)^alpha)
    end

    # Define Weibull hazard function h(t)
    # h(t) = (alpha/theta) * (t/theta)^(alpha-1)
    function weibull_hazard(t, alpha, theta)
        # alpha and theta are assumed > 0 from checks in the main loop
        if t < 0.0
            return 0.0 # Hazard is 0 before t=0
        elseif t == 0.0
            if alpha < 1.0       # Hazard is infinite at t=0 for alpha < 1
                return Inf
            elseif alpha == 1.0   # Hazard is constant 1/theta (Exponential distribution case)
                return 1.0 / theta
            else # alpha > 1.0       # Hazard is 0 at t=0 for alpha > 1
                return 0.0
            end
        else # t > 0.0
            return (alpha / theta) * (t / theta)^(alpha - 1.0)
        end
    end

    t_values = range(0.0, t_max, length = n_points)

    # Initialize plots
    # Margins can be added here if needed, e.g., margin=5Plots.mm
    plot_S = plot(
        xlabel = "Time (t)",
        ylabel = "Survival Probability S(t)",
        title = "Weibull Survival Function",
        legend = false,
    )

    plot_h = plot(
        xlabel = "Time (t)",
        ylabel = "Hazard Rate h(t)",
        title = "Weibull Hazard Function",
        legend = false,
        fontsize = 14,
    )
    # ylims can be set for plot_h if hazard values become very large, e.g. ylims=(0, desired_max_h)

    for params in param_sets
        alpha, theta = params[1], params[2]

        # Check for valid parameters (must be positive)
        if alpha <= 0.0 || theta <= 0.0
            @warn "Weibull parameters alpha (α) and theta (θ) must be positive. Skipping set (α=$alpha, θ=$theta)."
            continue
        end

        user_label_part = length(params) > 2 ? params[3] : ""

        # Calculate S(t) and h(t) values
        s_vals = [weibull_survival(t_val, alpha, theta) for t_val in t_values]
        h_vals = [weibull_hazard(t_val, alpha, theta) for t_val in t_values]

        # Create label for legend
        # Using round for cleaner display of float parameters in labels
        alpha_str = "α=$(round(alpha, sigdigits=3))"
        theta_str = "θ=$(round(theta, sigdigits=3))"

        if isempty(user_label_part)
            current_label = "$alpha_str, $theta_str"
        else
            current_label = "$user_label_part ($alpha_str, $theta_str)"
        end

        # Add curves to the plots
        # Plots.jl will cycle through default colors and line styles for each series
        plot!(plot_S, t_values, s_vals, label = current_label, linewidth = 1.5)
        plot!(plot_h, t_values, h_vals, label = current_label, linewidth = 1.5)
    end

    # Combine the two plots into a single figure with a 2x1 layout
    # Overall figure margins can be set here, e.g. left_margin=10Plots.mm
    combined_plot = plot(
        plot_S,
        plot_h,
        layout = (1, 2),
        size = plot_size,
        left_margin = 10mm,
        right_margin = 10mm,
        top_margin = 10mm,
        bottom_margin = 10mm,
        legend = :topright,
        legendfontsize = 14,
    )
    
    return combined_plot, plot_S # Return both the combined plot and the survival plot
end

# Example of how to use the function (can be uncommented and run in a suitable environment):
# ```julia
# # Ensure Plots.jl is loaded, e.g., using Plots
# 
# # Define parameter sets to plot: (alpha, theta, label)
params_to_visualize = [
    (1.0, 5.0, "Constant Hazard (α=1)"),      # alpha=1 (Exponential distribution)
    (2.0, 5.0, "Increasing Hazard (α>1)"), # alpha > 1
    (0.5, 5.0, "Decreasing Hazard (α<1)"), # alpha < 1
    (1.5, 3.0, "Characteristic Lifetime 3"),      # Varying theta
    (1.5, 7.0, "Characteristic Lifetime 7"),       # Varying theta
]
# 
# # Generate the plots
weibull_visualization, weibull_survival_plot_only = # Capture both returned plots
    plot_weibull_curves(params_to_visualize, t_max = 15.0, n_points = 300)
# 
# # To display the plot (e.g., in a REPL, Jupyter notebook, or Pluto.jl):
display(weibull_visualization)

##

## Juxtaposed KM and Weibull Plots

# Combine the Kaplan-Meier plot and the Weibull visualization into two subplots.
# p_full_km is the plot object for the Kaplan-Meier visualization.
# weibull_visualization is the plot object for the Weibull S(t) and h(t) curves.

juxtaposed_km_weibull = plot(
    p_full_km,
    weibull_visualization,
    layout = (1, 2),        # Arrange in 1 row, 2 columns
    size = (2200, 750),     # Overall size for the new combined plot; adjust as needed
    # You can add titles to these new "master" subplots if desired, e.g.:
    # plot_title = ["Observed Data with Kaplan-Meier" "Theoretical Weibull Curves"]
)

# Display the new combined plot
display(juxtaposed_km_weibull)

## Juxtaposed KM and Weibull Survival Curve Plot

# Combine the Kaplan-Meier plot and ONLY the Weibull survival curve into two subplots.
# p_full_km is the plot object for the Kaplan-Meier visualization.
# weibull_survival_plot_only is the standalone survival plot

juxtaposed_km_weibull_surv_only = plot(
    p_full_km,
    weibull_survival_plot_only, # Use the standalone survival plot
    layout = (1, 2),                   # Arrange in 1 row, 2 columns
    size = (1700, 700),                # Adjusted overall size for the new combined plot
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    # You can add titles to these new "master" subplots if desired, e.g.:
    # plot_title = ["Observed Data with Kaplan-Meier" "Theoretical Weibull Survival Curve S(t)"]
)

# Display the new combined plot
display(juxtaposed_km_weibull_surv_only)

## Stratified Kaplan-Meier Plot by Sex

# Create subsets for male and female
# Male: female .== false
# Female: female .== true

y_male = y[.!female]
event_male = event[.!female]

y_female = y[female]
event_female = event[female]

# Fit Kaplan-Meier estimators for each group
km_male_fit = fit(KaplanMeier, y_male, event_male)
km_female_fit = fit(KaplanMeier, y_female, event_female)

# Plot the stratified Kaplan-Meier curves
p_km_stratified_sex = plot(
    title = "Kaplan-Meier Survival Curves by Sex",
    xlabel = "Observed Time (days)",
    ylabel = "Survival Probability",
    legend = :topright, # or :best, :bottomright, etc.
    size = (800, 600),
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    ylims = (0, 1.05)
)

plot!(p_km_stratified_sex,
    km_male_fit.events.time,
    km_male_fit.survival,
    seriestype = :steppost,
    label = "Male",
    color = :steelblue, # Example color
    linewidth = 2
)

plot!(p_km_stratified_sex,
    km_female_fit.events.time,
    km_female_fit.survival,
    seriestype = :steppost,
    label = "Female",
    color = :coral, # Example color
    linewidth = 2
)

# Display the stratified plot
display(p_km_stratified_sex)



