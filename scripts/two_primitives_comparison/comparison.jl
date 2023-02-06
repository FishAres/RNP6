using DrWatson
@quickactivate "RNP6"
using LinearAlgebra, Statistics
using JLD2
using Plots
pyplot()

include(srcdir("gen_vae_utils.jl"))
## =====
function plot_ribbon(x; fribbon=std, kwargs...)
    plot(mean(x), ribbon=fribbon(x); kwargs...)
end

function plot_ribbon!(x; fribbon=std, kwargs...)
    plot!(mean(x), ribbon=fribbon(x); kwargs...)
end


function get_rec_losses_from_dict(model_dir)
    Ls = []
    for model in readdir(model_dir, join=false)
        # model_ind = m[findfirst(".jld2", m)[1]-1] |> string
        rec_losses = JLD2.load(joinpath(model_dir, model))["loss_dict"]["RLs"]
        push!(Ls, rec_losses[937:937:end]) # 937 mnist batches
    end
    return Ls
end

# bug where klqp and rec loss saved under wrong key
# model_dir = "saved_models/one_primitive_comparison/mnist_vanilla_VAE/loss_dicts"
# for model in readdir(model_dir, join=false)
#     loss_dict = JLD2.load(joinpath(model_dir, model))["loss_dict"]

#     RLs = loss_dict["KLs"]
#     KLs = loss_dict["RLs"]
#     loss_dict["RLs"] = RLs
#     loss_dict["KLs"] = KLs
#     JLD2.@save joinpath(model_dir, model) loss_dict
# end


begin

    model_dir_H = "saved_models/two_primitive_comparison/mnist_2lvl/loss_dicts"
    model_dir_vanilla = "saved_models/two_primitive_comparison/mnist_vanilla_VAE/loss_dicts"

    Ls_vanilla = get_rec_losses_from_dict(model_dir_vanilla)
    Ls_H = get_rec_losses_from_dict(model_dir_H)

    begin
        p = plot_ribbon(Ls_vanilla ./ 64,
            label="Conv VAE",
            foreground_color_legend=nothing,
            grid=false,
            thickness_scaling=1.5,)
        plot_ribbon!(Ls_H ./ 64, label="RNP 2-level")
        title!("Two primitives")
        ylabel!("MSE")
        xlabel!("Epoch")
    end
    savefig(p, "plots/primitive_comparison/two_primitives_RNP_VAE.pdf")
end

begin

    model_dir_H = "saved_models/one_primitive_comparison/mnist_2lvl/loss_dicts"
    model_dir_vanilla = "saved_models/one_primitive_comparison/mnist_vanilla_VAE/loss_dicts"

    Ls_vanilla = get_rec_losses_from_dict(model_dir_vanilla)
    Ls_H = get_rec_losses_from_dict(model_dir_H)

    begin
        p = plot_ribbon(map(x -> x[1:100], Ls_vanilla ./ 64),
            label="Conv VAE", foreground_color_legend=nothing)
        plot_ribbon!((Ls_H ./ 64), label="RNP 2-level")
        title!("One primitive")
        ylabel!("MSE")
        xlabel!("Epoch")
    end
    savefig(p, "plots/primitive_comparison/one_primitive_RNP_VAE.pdf")
end
