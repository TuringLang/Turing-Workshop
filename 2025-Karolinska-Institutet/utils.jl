using DynamicPPL
using MCMCChains
using Plots
using StatsPlots
using Survival

function plot_prior_pred_check(
    model::DynamicPPL.Model, 
    prior_samples::MCMCChains.Chains, 
    y,
    event
)    
    returned_values = returned(model, prior_samples)
    p_prior = plot(
        title = "Prior Predictive Check",
        xlabel = "Time (days)",
        ylabel = "Survival Probability",
        legend = :topright,
        size = (800, 600),
        left_margin = 10mm,
        right_margin = 10mm,
        top_margin = 10mm,
        bottom_margin = 10mm,
        ylims = (0, 1.05)
    )

    km_obs = fit(KaplanMeier, y, event)
    plot!(p_prior, 
        km_obs.events.time, 
        km_obs.survival,
        seriestype = :steppost,
        label = "Observed Data",
        color = :black,
        linewidth = 2
    )

    for returned_value in returned_values
        y_sim = returned_value.y
        km_sim = fit(KaplanMeier, y_sim, event)
        plot!(p_prior, 
            km_sim.events.time, 
            km_sim.survival,
            seriestype = :steppost,
            label = "",
            color = :lightblue,
            xlims = (0, maximum(y) * 1.1),
            alpha = 0.3
        )
    end
    display(p_prior)
end

function plot_posterior_pred_check(
    model::DynamicPPL.Model,
    posterior_samples::MCMCChains.Chains, 
    y, 
    event,
)
    p_post = plot(
        title = "Posterior Predictive Check",
        xlabel = "Time (days)",
        ylabel = "Survival Probability",
        legend = :topright,
        size = (800, 600),
        left_margin = 10mm,
        right_margin = 10mm,
        top_margin = 10mm,
        bottom_margin = 10mm,
        ylims = (0, 1.05)
    )

    km_obs = fit(KaplanMeier, y, event)
    plot!(p_post, 
        km_obs.events.time, 
        km_obs.survival,
        seriestype = :steppost,
        label = "Observed Data",
        color = :black,
        linewidth = 2
    )

    # Get parameter samples from the posterior
    params_chain = get(posterior_samples; section = :parameters)
    
    # Generate multiple posterior predictive samples
    n_samples = 100
    for i in 1:n_samples
        # Get a random sample index
        idx = rand(1:length(params_chain.α))
        
        # Fix the model with parameters from posterior
        fixed_model = fix(model, (; 
            α = params_chain.α[idx], 
            β = [params_chain.β[i][idx] for i in 1:size(X, 2)]
        ))
        
        # Generate simulated data
        sim_data = fixed_model()

        y_sim = sim_data.y

        km_sim = fit(KaplanMeier, y_sim, event)
        plot!(p_post, 
            km_sim.events.time, 
            km_sim.survival,
            seriestype = :steppost,
            label = "",
            color = :lightblue,
            xlims = (0, maximum(y) * 1.1),
            alpha = 0.3
        )
    end

    display(p_post)
end

function plot_posterior_pred_check_frailty(
    model::DynamicPPL.Model, 
    posterior_samples::MCMCChains.Chains, 
    y, 
    event,
)
    p_post = plot(
        # title = "Posterior Predictive Check",
        title = "Predictions vs Observed Data",
        xlabel = "Time (days)",
        ylabel = "Survival Probability",
        legend = :topright,
        size = (800, 600),
        left_margin = 10mm,
        right_margin = 10mm,
        top_margin = 10mm,
        bottom_margin = 10mm,
        ylims = (0, 1.05)
    )

    km_obs = fit(KaplanMeier, y, event)
    plot!(p_post, 
        km_obs.events.time, 
        km_obs.survival,
        seriestype = :steppost,
        label = "Observed Data",
        color = :black,
        linewidth = 2
    )
    
    params_chain = get(posterior_samples; section = :parameters)
    
    n_samples = 100
    for i in 1:n_samples
        idx = rand(1:length(params_chain))
        
        fixed_model = fix(model, (; 
            α = params_chain.α[idx], 
            β = [params_chain.β[i][idx] for i in 1:size(X, 2)],
            k = params_chain.k[idx],
            v = [params_chain.v[i][idx] for i in 1:size(X, 1)]
        ))
        
        sim_data = fixed_model()
        y_sim = sim_data.y

        km_sim = fit(KaplanMeier, y_sim, event)
        plot!(p_post, 
            km_sim.events.time, 
            km_sim.survival,
            seriestype = :steppost,
            label = "",
            color = :lightblue,
            xlims = (0, maximum(y) * 1.1),
            alpha = 0.3
        )
    end

    display(p_post)
end

