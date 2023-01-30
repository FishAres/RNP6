using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using JLD2
using Plots

## =====

model_dir = "saved_models/systematic_n_filters/mnist/"

begin
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
end

begin
    better_Ldict = Dict()
    n_filters = string.([8, 16, 32, 64])
    for (i, dict) in enumerate(keys(Ldict))
        D = Ldict[dict]
        Ls = []
        for ld in keys(D)
            push!(Ls, [L[end] for L in D[ld]["RLs"]])
        end
        better_Ldict[n_filters[i]] = Ls
    end
end


begin
    p = plot()
    [plot!(mean(L), ribbon=std(L), label=n_filters[i], grid=false, foreground_color_legend=nothing,
    legendtitle="# filters") for (i, L) in enumerate(values(better_Ldict))]
    p
end
