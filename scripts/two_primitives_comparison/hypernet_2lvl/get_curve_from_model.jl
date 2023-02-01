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
    :π => 64, # latent dim size
    :img_channels => 1, # number of image channels
    :esz => 32, # RNN input size
    :ha_sz => 32, # policy RNN hidden size
    :α => 1.0f0, # reconstruction loss weight
    :β => 0.1f0, # KL loss weight
    :add_offset => true, # add scaling offset at each level
    :fa_out => identity, # f_policy RNN activation function 
    :f_z => elu, # f_state RNN activation function
    :asz => 6, # number of spatial transformer params (scale x + y, bias x + y, rotation, shear)
    :seqlen => 5, # number of patches 
    :λ => 1.0f-3, # hypernet regularization weight
    :λpatch => Float32(1 / 4), # regularization for z1 decoding (patch) loss
    :scale_offset => 1.6f0, # offset for scaling at each level
    :D => Normal(0.0f0, 1.0f0)) # prior distribution
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

modelpath = "saved_models/hypernet_2lvl/mnist_2lvl_inc_λpatch/a_sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=1_imzprod=784_scale_offset=2.4_seqlen=4_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.16_π=16_50eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## +====

function sample_(z, x; args=args)
    x̂s = []
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push!(x̂s, x̂ |> cpu)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push!(x̂s, x̂ |> cpu)
    end
    return out |> cpu, x̂s
end

z = randn(Float32, args[:π], args[:bsz]) |> gpu
x = first(test_loader)

out, patches = sample_(z, x)

# hackily find a curve you like
i = 0
begin
    i = mod(i + 1, args[:bsz])
    a = reshape(patches[2][:, i], 28, 28)
    plot_digit(a)
    title!(string(i))
end

curve_patch = patches[2][:, 21]
curve_patch = repeat(curve_patch, 1, args[:bsz])
JLD2.@save "saved_models/two_primitive_comparison/curve_primitives/curve1.jld2" curve_patch
