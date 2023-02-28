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

include(srcdir("gen_vae_utils_larger.jl"))

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

device!(0)

dev = gpu

##=====


train_digits, train_labels = Omniglot(; split=:train)[:]
test_digits, test_labels = Omniglot(; split=:test)[:]

train_digits = imresize(train_digits, (28, 28))
a = (1.0f0 .- train_digits[:, :, 2])
using ImageTransformations
center(a)

all_chars = load("../Recur_generative/data/exp_pro/omniglot_train.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
num_train = trunc(Int, 0.9 * length(xs))

new_chars = load("../Recur_generative/data/exp_pro/omniglot_eval.jld2")
xs_new = shuffle(vcat((new_chars[key] for key in keys(new_chars))...))

function fast_cat(xs)
    x_array = zeros(Float32, size(xs[1])..., length(xs))
    Threads.@threads for i in 1:length(xs)
        x_array[:, :, i] = xs[i]
    end
    return x_array
end

xs_cat = fast_cat(xs)
train_chars = xs_cat[:, :, 1:num_train]
val_chars = xs_cat[:, :, (num_train+1):end]

new_xs_cat = fast_cat(xs_new)

train_loader = DataLoader(dev(train_chars); batchsize=args[:bsz], shuffle=true,
    partial=false)
val_loader = DataLoader(dev(val_chars); batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(dev(new_xs_cat); batchsize=args[:bsz], shuffle=true, partial=false)


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


## =====
args[:π] = 32
args[:D] = Normal(0.0f0, 1.0f0)

mEnc_za_z = Chain(
    HyDense(args[:π] + args[:asz], 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:esz], args[:bsz], elu),
    flatten,
)

l_enc_za_z = get_param_sizes(mEnc_za_z)
# l_enc_za_z = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:esz], args[:π]) # μ, logvar

mdec = Chain(HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, 784, args[:bsz], relu),
    flatten,
)

l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z...; l_fx; l_dec_x...]

mEnc_za_a = Chain(
    HyDense(args[:π] + args[:asz], 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:esz], args[:bsz], elu),
    flatten,
)
l_enc_za_a = get_param_sizes(mEnc_za_a)
# l_enc_za_a = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:esz], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a...; l_fa; l_dec_a]

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
        BasicBlock(32 => 32, +),
        flatten)
    outsz = Flux.outputsize(enc1,
        (args[:img_size]..., args[:img_channels],
            args[:bsz]))
    Chain(enc1,
        Dense(outsz[1], 128),
        LayerNorm(128, elu),
        Dense(128, 128),
        LayerNorm(128, elu),
        Dense(128, 128),
        LayerNorm(128, elu),
        Dense(128, 128),
        LayerNorm(128, elu),
        Split(Dense(128, args[:π]), Dense(128, args[:π])))
end)

ps = Flux.params(Hx, Ha, Encoder)

## ====

x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)
let
    p = plot_recs(x, inds)
    display(p)
end
## =====
save_folder = "hypernet_2lvl_larger_za_z_enc"
alias = "omni_2lvl"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 3
args[:scale_offset] = 1.8f0
args[:λ] = 1.0f-6
args[:λpatch] = 0.0f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.1f0

args[:η] = 1e-4

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls = []
    for epoch in 65:200
        if epoch % 50 == 0
            opt.eta = 0.67 * opt.eta
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
            psamp = stack_ims(out) |> plot_digit
            log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)
        end

        Lval = test_model(val_loader)
        log_value(lg, "test loss", Lval)
        @info "Test loss: $Lval"
        if epoch % 50 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
        push!(Ls, ls)
    end
end
## =====

begin
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = sample_(z, x)
    psamp = stack_ims(out) |> plot_digit
end


## =====

function get_μ_zs(x)
    z1s = []
    μ, logvar = Encoder(x)
    θsz = Hx(μ)
    θsa = Ha(μ)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push!(z1s, cpu(z1))
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push!(z1s, cpu(z1))
    end
    return μ |> cpu, z1s
end

## === TSne
using TSne, Clustering

begin
    z2s, z1s = [], []
    for (i, x) in enumerate(test_loader)
        μ, z1 = get_μ_zs(x)
        push!(z2s, μ)
        push!(z1s, hcat(z1...))
    end
end

function shuffle_batchwise(x1, x2)
    b = shuffle([x1; x2])
    return hcat(b...)
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


Z = shuffle_batchwise(z2s, z1s)
sample_inds = sample(1:size(Z, 2), 4000, replace=false)


Ys = tsne(Z[:, sample_inds]')
scatter(Ys[:, 1], Ys[:, 2])

n_clusters = 40
R = kmeans(Ys', n_clusters)
inds = assignments(R)

begin
    p = plot(legend=false)
    for i in 1:n_clusters
        scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
    end
    p
end

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

outputs[1]
out_cat = cat(outputs..., dims=3)
out_cat_x̂ = cat(outputs_x̂..., dims=3)
trunc_inds = inds[1:size(out_cat, 3)]

ind = 0
begin
    ind += 1
    b = out_cat[:, :, trunc_inds.==ind]
    # b = out_cat_x̂[:, :, trunc_inds.==ind]
    stack_ims(b, n=8) |> plot_digit
end