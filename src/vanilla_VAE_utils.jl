using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Distributions
using ProgressMeter
using ProgressMeter: Progress
using Plots

include(srcdir("nn_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))
## =========

sample_z(μ, logvar, r) = μ + r .* (exp.(logvar))

kl_loss(μ, logvar) = sum(@. (exp(logvar) + μ^2 - logvar - 1.0f0))

function model_loss(x; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, rand(args[:D], args[:π], args[:bsz]) |> gpu)
    out = Decoder(z)
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss, klqp
end

function sample_(z)
    out = Decoder(z)
    out |> cpu
end

function plot_recs(x, inds; args=args)
    function plot_rec(pred, x, ind)
        p1 = plot_digit(cpu(pred)[:, :, 1, ind])
        p2 = plot_digit(cpu(x)[:, :, ind])
        plot(p1, p2)
    end
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, rand(args[:D], args[:π], args[:bsz]) |> gpu)
    out = sample_(z)
    p = [plot_rec(out, x, ind) for ind in inds]
    plot(p..., layout=(length(inds), 1))
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    rec_losses, KLs = [], []
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                return log_value(lg, "KL loss", klqp)
            end
            Zygote.ignore() do 
                push!(rec_losses, rec_loss)
                push!(KLs, klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * norm(Flux.params(Decoder))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses, rec_losses, KLs
end

function test_model(test_data; D=args[:D])
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(x)
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

function stack_ims(xs; n=8)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, x) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    return x_
end
