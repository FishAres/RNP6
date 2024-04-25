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
using MLUtils
include(srcdir("eth80_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(:bsz => 64,
    :img_size => (50, 50),
    :π => 64, # latent dim size
    :img_channels => 3,
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

## =====

eth80_train = load(datadir("exp_pro", "eth80_segmented_train.jld2"))["eth80_train"]
eth80_test = load(datadir("exp_pro", "eth80_segmented_test.jld2"))["eth80_test"]


train_loader = DataLoader((dev(eth80_train)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(eth80_test)); batchsize=args[:bsz], shuffle=true,
    partial=false)

## =====

const dev = has_cuda() ? gpu : cpu

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
    # xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=4))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end


function imview_cifar(x)
    colorview(RGB, permutedims(batched_adjoint(x), [3, 1, 2]))
end

function plot_rec_cifar(x, out, xs::Vector, ind)
    out_ = reshape(cpu(out), args[:img_size]..., 3, :)
    x_ = reshape(cpu(x), args[:img_size]..., 3, size(x)[end])
    p1 = plot(imview_cifar(out_[:, :, :, ind]), axis=nothing,)
    p2 = plot(imview_cifar(x_[:, :, :, ind]), axis=nothing, size=(20, 20))
    p3 = plot([plot(imview_cifar(x[:, :, :, ind]), axis=nothing) for x in xs]...)
    return plot(p1, p2, p3, layout=(1, 3))
end

function plot_recs(x, inds; plot_seq=true, args=args)
    full_recs, patches, xys, patches_t = get_loop(x)
    p = if plot_seq
        let
            patches_ = map(x -> reshape(x, args[:img_size]..., args[:img_channels], size(x)[end]), patches)
            [plot_rec_cifar(full_recs[end], x, patches_, ind) for ind in inds]
        end
    else
        [plot_rec_cifar(full_recs[end], x, ind) for ind in inds]
    end

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
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


mdec = Chain(
    HyDense(args[:π], 400, args[:bsz], elu),
    x -> reshape(x, 10, 10, 4, :),
    HyConvTranspose((5, 5), 4 => 32, args[:bsz], relu, stride=1),
    HyConvTranspose((4, 4), 32 => 32, args[:bsz], relu, stride=2, pad=2),
    HyConvTranspose((4, 4), 32 => 3, args[:bsz], relu, stride=2, pad=2)
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

## ======

modelpath = "saved_models/hypernet_2lvl_larger_za_z_enc/eth80_2lvl/a_sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=3_imzprod=2500_scale_offset=2.4_seqlen=4_α=1.0_β=0.1_η=0.0001_λ=1e-6_λpatch=0.049_π=32_200eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## =====
args[:seqlen] = 4
args[:scale_offset] = 2.4f0

x = first(test_loader)
begin
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = sample_(z, x)[:, :, :, 1:16]
    psamp = stack_ims(out, n=4) |> imview_cifar |> x -> imresize(x, (400, 400))
end