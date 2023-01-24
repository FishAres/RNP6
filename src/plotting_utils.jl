using Plots

function plot_digit(x; color=:grays, alpha=1, boundc=true, kwargs...)
    return heatmap(x[:, end:-1:1]';
                   color=color,
                   clim=boundc ? (0, 1) : (minimum(x), maximum(x)),
                   axis=nothing,
                   colorbar=false,
                   alpha=alpha,
                   kwargs...)
end

function plot_digit!(x; color=:grays, alpha=1, boundc=true, kwargs...)
    return heatmap!(x[:, end:-1:1]';
                    color=color,
                    clim=boundc ? (0, 1) : (minimum(x), maximum(x)),
                    # clim=(0, 1),
                    axis=nothing,
                    colorbar=false,
                    alpha=alpha,
                    kwargs...)
end
