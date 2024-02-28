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
args = Dict(
    :bsz => 64, # batch size
    :img_size => (28, 28),
    :π => 8, # latent dim size / state RNN hidden dim size
    :img_channels => 1,
    :esz => 32, # RNN input size
    :ha_sz => 32, # policy RNN hidden size
    :f_z => elu, # state RNN activation
    :α => 1.0f0, # rec. loss weighting
    :β => 0.1f0, # KL weighting
    :add_offset => true, # add scale offset to affine transforms
    :asz => 6, # of spatial transformer params
    :seqlen => 3, # RNP sequence length
    :lr => 1e-4, # learning rate
    :λ => 1.0f-3, # hypernet regularization weight
    :λpatch => Float32(1 / 4), # patch reconstruction regularization weight
    :scale_offset => 1.6f0, # affine transform scale offset
)
args[:imzprod] = prod(args[:img_size])

## =====

# device!(0) # specify gpu if there are multiple
const dev = has_cuda() ? gpu : cpu

## =====

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]

train_loader = DataLoader((dev(train_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)

## =====

const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))


## =====

mEnc_za_z = Chain(
    HyDense(args[:π] + args[:asz], 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:esz], args[:bsz], elu),
    flatten,
)

l_enc_za_z = get_param_sizes(mEnc_za_z)
l_fx = get_rnn_θ_sizes(args[:esz], args[:π])

mdec = Chain(HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, 784, args[:bsz], relu),
    flatten,
)

l_dec_x = get_param_sizes(mdec)

const Hstate_bounds = [l_enc_za_z...; l_fx; l_dec_x...]

mEnc_za_a = Chain(
    HyDense(args[:π] + args[:asz], 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:esz], args[:bsz], elu),
    flatten,
)
l_enc_za_a = get_param_sizes(mEnc_za_a)
l_fa = get_rnn_θ_sizes(args[:esz], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

const Hpolicy_bounds = [l_enc_za_a...; l_fa; l_dec_a]

## =====
H_state = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hstate_bounds) + args[:π], bias=false),
) |> gpu

H_policy = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hpolicy_bounds) + args[:asz]; bias=false)
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

ps = Flux.params(H_state, H_policy, Encoder)
## ====

x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)
let
    p = plot_recs(x, inds)
    display(p)
end
## =====
save_folder = "RNP_2lvl_mnist"
save_dir = get_save_dir(save_folder, "")

## =====
args[:seqlen] = 3
args[:scale_offset] = 1.9f0
args[:λ] = 1.0f-6
args[:λpatch] = 0.0f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.1f0

args[:η] = 1e-4

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)


## ====
function train_model(opt, ps, train_loader, test_loader; n_epochs=10, logger=nothing, clip_grad=False, sample_every=5)
    Ls = []
    for epoch in 1:n_epochs
        if epoch % 5 == 0
            args[:λpatch] = max(args[:λpatch] + 1.0f-3, 1.0f-2)
            log_value(logger, "λpatch", args[:λpatch])
        end

        ls = train_epoch(opt, ps, train_loader; epoch=epoch, logger=logger)

        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        logger !== nothing && log_image(logger, "recs_$(epoch)", p)

        if epoch % sample_every == 0
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = sample_(z, x)
            psamp = stack_ims(out) |> plot_digit
            log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)
        end

        Lval = test_model(test_loader)
        log_value(logger, "test loss", Lval)
        @info "Test loss: $Lval"
        if epoch == n_epochs
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
        push!(Ls, ls)
    end
end

save_model((H_state, H_policy, Encoder), joinpath(save_dir, savename(args) * "_$(100)eps"))

modelpath = "saved_models/RNP_2lvl_mnist/add_offset=true_asz=6_bsz=64_esz=32_ha_sz=32_img_channels=1_imzprod=784_scale_offset=1.9_seqlen=3_α=1.0_β=0.1_η=0.0001_λ=1e-6_λpatch=0.0_π=8_100eps.bson"
H_state, H_policy, Encoder = load(modelpath)[:model] |> gpu

## =====

x = first(test_loader)
begin
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = sample_(z, x)
    psamp = stack_ims(out) |> plot_digit
end


function full_sequence_patches(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    patches = []
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    trans_patch = sample_patch(x̂, a1, sampling_grid)
    out = trans_patch
    push!(patches, out |> cpu)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        trans_patch = sample_patch(x̂, a1, sampling_grid)
        out += trans_patch
        push!(patches, out |> cpu)
    end
    return patches, out
end

function full_sequence_patches(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence_patches(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

function get_loop_patches(x; args=args)
    outputs = recs, comb_patches, patches, trans_patches, as, patches_t = [], [], [], [], [], [], []
    r = gpu(rand(args[:D], args[:π], args[:bsz]))
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    small_patches, out_small = full_sequence_patches(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, small_patches, out_small, out, a1, patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        small_patches, out_small = full_sequence_patches(z1, patch_t)
        trans_patch = sample_patch(out_small, a1, sampling_grid)
        out += trans_patch
        push_to_arrays!((out, small_patches, out_small, trans_patch, a1, patch_t), outputs)
    end
    return outputs
end

function sample_patches(z, x; args=args)
    outputs = recs, comb_patches, patches, trans_patches, as, patches_t = [], [], [], [], [], [], []
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    small_patches, out_small = full_sequence_patches(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, small_patches, out_small, out, a1, patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        small_patches, out_small = full_sequence_patches(z1, patch_t)
        trans_patch = sample_patch(out_small, a1, sampling_grid)
        out += trans_patch
        push_to_arrays!((out, small_patches, out_small, trans_patch, a1, patch_t), outputs)
    end
    return outputs
end



begin
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    recs, small_patches, patches, trans_patches, as, patches_t = sample_patches(z, x)
    a = [imresize(view_patches_rgb(trans_patches, ind), (60, 60)) for ind in 1:16]
    a = map(x -> collect(x'), a)
    stack_ims(cat(a..., dims=3); n=4)
end
