## Data

using DataFrames, RDatasets
using Statistics: mean, std
using Plots.PlotMeasures

# Load and prepare data
df_lung = dataset("survival", "cancer")
df = dropmissing(df_lung, [:Time, :Status, :Age, :Sex])

# Standardize Age and create features
standardise(x) = (x .- mean(x)) ./ std(x)
age_std = standardise(df.Age)
female = df.Sex .== 2
X = hcat(ones(nrow(df)), age_std, female)

# Extract outcome variables
y = Float64.(df.Time)
event = df.Status .== 2  # Status=1 -> censored, Status=2 -> dead

## Kaplan-Meier

using StatsPlots, Plots, Survival

# Fit Kaplan-Meier estimator
km_overall_fit = fit(KaplanMeier, y, event)

# Sort data for visualization
sort_indices = sortperm(y)
y_sorted = y[sort_indices]
event_sorted = event[sort_indices]
indices_sorted = 1:length(y)

# Plot data with KM curve
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
    yflip = true,
)

# Add KM curve on secondary y-axis
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
    legend = :bottomleft,
    ylims = (0, 1.05),
    left_margin = 20mm,
    right_margin = 30mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    size = (800, 600),
)

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
    yflip = true,
    size = (800, 600),
    left_margin = 20mm,
    right_margin = 30mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
)

##

"""
Generates plots for Weibull survival S(t) and hazard h(t) functions.

Args:
    param_sets: List of tuples (alpha, theta, label_str)
    t_max: Maximum time value for x-axis
    n_points: Number of points to calculate
    plot_size: Size of the combined plot

Returns:
    Combined plot object and survival plot
"""
function plot_weibull_curves(
    param_sets;
    t_max = 10.0,
    n_points = 200,
    plot_size = (1600, 800),
)
    function weibull_survival(t, alpha, theta)
        if t < 0.0
            return 1.0
        end
        return exp(-(t / theta)^alpha)
    end

    function weibull_hazard(t, alpha, theta)
        if t < 0.0
            return 0.0
        elseif t == 0.0
            if alpha < 1.0
                return Inf
            elseif alpha == 1.0
                return 1.0 / theta
            else
                return 0.0
            end
        else
            return (alpha / theta) * (t / theta)^(alpha - 1.0)
        end
    end

    t_values = range(0.0, t_max, length = n_points)

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

    for params in param_sets
        alpha, theta = params[1], params[2]

        if alpha <= 0.0 || theta <= 0.0
            @warn "Weibull parameters alpha (α) and theta (θ) must be positive. Skipping set (α=$alpha, θ=$theta)."
            continue
        end

        user_label_part = length(params) > 2 ? params[3] : ""

        s_vals = [weibull_survival(t_val, alpha, theta) for t_val in t_values]
        h_vals = [weibull_hazard(t_val, alpha, theta) for t_val in t_values]

        alpha_str = "α=$(round(alpha, sigdigits=3))"
        theta_str = "θ=$(round(theta, sigdigits=3))"

        current_label =
            isempty(user_label_part) ? "$alpha_str, $theta_str" :
            "$user_label_part ($alpha_str, $theta_str)"

        plot!(plot_S, t_values, s_vals, label = current_label, linewidth = 1.5)
        plot!(plot_h, t_values, h_vals, label = current_label, linewidth = 1.5)
    end

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

    return combined_plot, plot_S
end

params_to_visualize = [
    (1.0, 5.0, "Constant Hazard (α=1)"),
    (2.0, 5.0, "Increasing Hazard (α>1)"),
    (0.5, 5.0, "Decreasing Hazard (α<1)"),
    (1.5, 3.0, "Characteristic Lifetime 3"),
    (1.5, 7.0, "Characteristic Lifetime 7"),
]

weibull_visualization, weibull_survival_plot_only =
    plot_weibull_curves(params_to_visualize, t_max = 15.0, n_points = 300)

display(weibull_visualization)

##

## Juxtaposed KM and Weibull Plots

juxtaposed_km_weibull =
    plot(p_full_km, weibull_visualization, layout = (1, 2), size = (2200, 750))

display(juxtaposed_km_weibull)

## Juxtaposed KM and Weibull Survival Curve Plot

juxtaposed_km_weibull_surv_only = plot(
    p_full_km,
    weibull_survival_plot_only,
    layout = (1, 2),
    size = (1700, 700),
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
)

display(juxtaposed_km_weibull_surv_only)

## Stratified Kaplan-Meier Plot by Sex

y_male = y[.!female]
event_male = event[.!female]

y_female = y[female]
event_female = event[female]

km_male_fit = fit(KaplanMeier, y_male, event_male)
km_female_fit = fit(KaplanMeier, y_female, event_female)

p_km_stratified_sex = plot(
    title = "Kaplan-Meier Survival Curves by Sex",
    xlabel = "Observed Time (days)",
    ylabel = "Survival Probability",
    legend = :topright,
    size = (800, 600),
    left_margin = 10mm,
    right_margin = 10mm,
    top_margin = 10mm,
    bottom_margin = 10mm,
    ylims = (0, 1.05),
)

plot!(
    p_km_stratified_sex,
    km_male_fit.events.time,
    km_male_fit.survival,
    seriestype = :steppost,
    label = "Male",
    color = :steelblue,
    linewidth = 2,
)

plot!(
    p_km_stratified_sex,
    km_female_fit.events.time,
    km_female_fit.survival,
    seriestype = :steppost,
    label = "Female",
    color = :coral,
    linewidth = 2,
)

display(p_km_stratified_sex)
