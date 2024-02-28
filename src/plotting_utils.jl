using Plots

"""
Convenience function for plotting a heatmap in grayscale and the right orientation
    to view a digit
"""
function plot_digit(x; color=:grays, alpha=1, boundc=true, kwargs...)
    return heatmap(x[:, end:-1:1]';
        color=color,
        clim=boundc ? (0, 1) : (minimum(x), maximum(x)),
        axis=nothing,
        colorbar=false,
        alpha=alpha,
        kwargs...)
end

"""
In-place version of `plot_digit`
"""
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

"""
Stack the arrays in `xs` into an `n x n` grid
"""
function stack_ims(xs; n=nothing)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end