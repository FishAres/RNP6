using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using JLD2
using Plots

## =====

# model_dir = "saved_models/systematic_n_filters/vanilla_VAE/mnist"
model_dir = "saved_models/systematic_n_filters/GRU_recursive/"

function parse_loss_dict(model_dir)
    Ldict = Dict()
    [
        begin
            Ldict[nf] = Dict()
        end for nf in ["8", "16", "32", "64"]
    ]
    for model in readdir(model_dir, join=false)
        tmp_inds = findfirst("_", model)
        n_filters = model[1:tmp_inds[end]-1]
        model_ind_loc = findfirst("model_ind", model)[end] + 2
        model_ind = model[model_ind_loc] |> string
        Ldict["$n_filters"][model_ind] = JLD2.load(joinpath(model_dir, model))["loss_dict"]
    end
    return Ldict
end

function get_better_Ldict(Ldict; key="RLs")
    n_mnist_batches = 937
    better_Ldict = Dict()
    for (i, dict) in enumerate(keys(Ldict))
        D = Ldict[dict]
        Ls = []
        for ld in keys(D)
            push!(Ls, D[ld][key][1:n_mnist_batches:end])
        end
        better_Ldict[dict] = Ls
    end
    filter!(x -> !isempty(x.second), better_Ldict)
    return better_Ldict
end


better_Ldict = get_better_Ldict(parse_loss_dict(joinpath(model_dir, "GRU")))
better_Ldict_LSTM = get_better_Ldict(parse_loss_dict(joinpath(model_dir, "LSTM")))



function smoothed_loss(better_Ldict, n_filters; window=10)
    nf = string(n_filters)
    [mean(L[end-window:end]) for L in collect(values(better_Ldict[nf]))]
end


LSTM_losses = [smoothed_loss(better_Ldict, nf) for nf in [16, 32, 64]]
scatter(1:3, LSTM_losses, legend=false)


function plot_ribbon(x; fribbon=std, kwargs...)
    plot(mean(x), ribbon=fribbon(x); kwargs...)
end

function plot_ribbon!(x; fribbon=std, kwargs...)
    plot!(mean(x), ribbon=fribbon(x); kwargs...)
end

function plot_loss_curves(Ldict)
    p = plot()
    [plot!(mean(L), ribbon=std(L), label=n_filters[i], grid=false, foreground_color_legend=nothing,
        legendtitle="# filters") for (i, L) in enumerate(values(Ldict))]
    p
end

function plot_loss_curves!(Ldict)
    [plot!(mean(L), ribbon=std(L), label=n_filters[i], grid=false, foreground_color_legend=nothing,
        legendtitle="# filters") for (i, L) in enumerate(values(Ldict))]
end

