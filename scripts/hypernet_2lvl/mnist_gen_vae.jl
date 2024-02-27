using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(:bsz => 64,
    :img_size => (28, 28),
    :π => 64, # latent dim size
    :img_channels => 1,
    :esz => 32, # RNN input size
    :ha_sz => 32, # policy RNN hidden size
    :α => 1.0f0,
    :β => 0.1f0,
    :add_offset => true,
    :fa_out => identity,
    :f_z => elu,
    :asz => 6, # of spatial transformer params
    :glimpse_len => 4,
    :seqlen => 5,
    :λ => 1.0f-3, # hypernet regularization weight
    :λpatch => Float32(1 / 4),
    :a_sample_len => 8,
    :scale_offset => 1.6f0,
    :D => Normal(0.0f0, 1.0f0))
args[:imzprod] = prod(args[:img_size])

## =====

device!(1)

dev = gpu

##=====

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((dev(train_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)

## =====

dev = has_cuda() ? gpu : cpu

const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))

## ====

function sample_(z, x; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return out |> cpu
end


function stack_ims(xs; n=8)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end

function sample_levels(z, x; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    out_x̂ = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        out_x̂ += sample_patch(x̂, a1, sampling_grid)
    end
    return out |> cpu, out_x̂ |> cpu
end

function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(out_small); agg=sum)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(out_small); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end
## =====
args[:π] = 16
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:esz], args[:π]) # μ, logvar

mdec = Chain(HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, 784, args[:bsz], relu),
    flatten,
)

l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

l_enc_za_a = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:esz], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

Hx = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz]; bias=false)
) |> gpu

# init_hyper!(Hx)
# init_hyper!(Ha)

Encoder = gpu(let
    enc1 = Chain(x -> reshape(x, args[:img_size]..., args[:img_channels], :),
        Conv((5, 5), args[:img_channels] => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        flatten)
    outsz = Flux.outputsize(enc1,
        (args[:img_size]..., args[:img_channels],
            args[:bsz]))
    Chain(enc1,
        Dense(outsz[1], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Split(Dense(64, args[:π]), Dense(64, args[:π])))
end)

ps = Flux.params(Hx, Ha, Encoder)

## ====
x = first(test_loader)
let
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(x, inds)
    display(p)
end
## =====
save_folder = "hypernet_2lvl"
alias = "mnist_2lvl_inc_λpatch_z1_gz0"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 4
args[:scale_offset] = 2.4f0
args[:λ] = 1.0f-6
args[:λpatch] = 0.0f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.5f0

args[:η] = 1e-4

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
log_value(lg, "lambda_patch", args[:λpatch])
## ====
begin
    Ls = []
    for epoch in 1:200
        if epoch % 25 == 0
            opt.eta = 0.9 * opt.eta
            log_value(lg, "learning_rate", opt.eta)

        end

        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)

        if epoch % 5 == 0
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = sample_(z, x)
            out, out_x̂ = sample_levels(z, x)
            psamp_out = plot_digit(stack_ims(out), title="2 level")
            psamp_x̂ = plot_digit(stack_ims(out_x̂), title="1 level")
            psamp = plot(psamp_out, psamp_x̂)
            log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)

            λpatch_lim = Float32(1 / args[:seqlen])
            args[:λpatch] = min(args[:λpatch] + 2.0f-3, λpatch_lim)
            log_value(lg, "lambda_patch", args[:λpatch])
        end

        Lval = test_model(test_loader)
        log_value(lg, "test loss", Lval)
        @info "Test loss: $Lval"
        if epoch % 50 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
        push!(Ls, ls)
    end
end
## =====


function full_sequence_z0s(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    z0s = []
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    push!(z0s, cpu(z1))
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
        push!(z0s, cpu(z1))
    end
    return out, z0s
end

function full_sequence_z0s(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence_z0s(models, z0, a0, x; args=args, scale_offset=scale_offset)
end


function get_μ_zs(x)
    z1s, z0s = [], []
    μ, logvar = Encoder(x)
    θsz = Hx(μ)
    θsa = Ha(μ)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small, z0 = full_sequence_z0s(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push!(z1s, cpu(z1))
    push!(z0s, cpu(z0))
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small, z0 = full_sequence_z0s(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push!(z1s, cpu(z1))
        push!(z0s, cpu(z0))
    end
    return μ |> cpu, z1s, z0s
end

## === TSne
using TSne, Clustering

begin
    z2s, z1s, z0s = [], [], []
    for (i, x) in enumerate(test_loader)
        μ, z1, z0 = get_μ_zs(x)
        push!(z2s, μ)
        push!(z1s, hcat(z1...))
        push!(z0s, hcat(z0...))
    end
end

function shuffle_batchwise(x1, x2)
    b = shuffle([x1; x2])
    return hcat(b...)
end

z0s_stacked = hcat(map(x -> hcat(x...), z0s)...)
z1s_stacked = hcat(z1s...)
z2s_stacked = hcat(z2s...)

all_zs = hcat(shuffle(collect(eachcol(hcat([z0s_stacked, z1s_stacked, z2s_stacked]...))))...)

z1_z0_samples = hcat([z1s_stacked, z0s_stacked]...)
z1_z0_inds = sample(1:size(z1_z0_samples, 2), 5120)

Ys = tsne(z1_z0_samples[:, z1_z0_inds]')
scatter(Ys[:, 1], Ys[:, 2])



## ============


function get_cat_z_outputs(Z, sample_inds; args=args)
    x = first(test_loader)
    a = batch.(collect(partition(eachcol(Z[:, sample_inds]), args[:bsz]))) |> gpu
    outputs, outputs_x̂ = let
        out_, out_x̂_ = [], []
        for z in a
            out, out_x̂ = sample_levels(z, x) |> cpu
            push!(out_, dropdims(out, dims=3))
            push!(out_x̂_, dropdims(out_x̂, dims=3))
        end
        out_, out_x̂_
    end
    return outputs, outputs_x̂
end

function get_nearest_point(Ys, R::KmeansResult, cluster_ind)
    centroid = argmin(sqrt.((Ys[:, 1] .- R.centers[1, cluster_ind]) .^ 2) + sqrt.((Ys[:, 2] .- R.centers[2, cluster_ind]) .^ 2))
    return centroid
end

function get_centroid_image(Ys, R, Z, cluster_ind; plot=false)
    x = first(test_loader)
    zi = get_nearest_point(Ys, R, cluster_ind)
    Zi = repeat(Z_[:, zi], 1, args[:bsz]) |> gpu
    out, out_x̂ = sample_levels(Zi, x) |> cpu
    if plot
        return plot_digit(out_x̂[:, :, 1, 1])
    else
        return out_x̂[:, :, 1, 1]
    end
end

begin
    n_clusters = 45
    R = kmeans(Ys', n_clusters)
    inds = assignments(R)
    cluster_ims = [get_centroid_image(Ys, R, Z, k) for k in 1:n_clusters]
end

begin
    p = plot(
        xlim=(-120, 120),
        ylim=(-120, 120),
        axis=nothing,
        xaxis=false,
        yaxis=false,
        legend=false,
        size=(800, 800),
    )

    for i in 1:n_clusters
        scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
    end

    imsize = (12, 12)
    imctr = imsize ./ 2
    for i in 1:n_clusters
        x, y = R.centers[:, i] .+ 7
        heatmap!(
            x-imctr[1]+1:x+imctr[1],
            y-imctr[2]+1:y+imctr[2],
            imresize(cluster_ims[i], 12, 12),
            color=:grays,
            clim=(0, 1),
            alpha=0.9,
            axis=nothing,
            grid=false,
        )
    end
    # savefig(p, "plots/mnist_tsne/cluster_and_ims_reference.png")
    p
end
## =====
begin
    p = plot(
        xlim=(-120, 120),
        ylim=(-120, 120),
        axis=nothing,
        # xaxis=false,
        # yaxis=false,
        legend=false,
        size=(800, 800),
        # xlabel="Dimension 1",
        # ylabel="Dimension 2",
    )

    for i in 1:n_clusters
        scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
        # scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
    end
    savefig(p, "plots/mnist_tsne/tsne_clusters.png")
    p
end

begin
    for i in 1:n_clusters
        p = plot_digit(
            imresize(cluster_ims[i], 12, 12),
            clim=(0, 1),
            alpha=0.9,
            axis=nothing,
            colorbar=false,
            color=:grays,
        )
        # savefig(p, "plots/mnist_tsne/cluster#$i.png")
        p
    end
end
