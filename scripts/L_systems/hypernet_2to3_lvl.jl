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
using ArgParse

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, # batch size
    :img_size => (50, 50),
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
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--device_id"
        help = "GPU device"
        arg_type = Int
        default = 0
        "--model_ind"
        arg_type = Int
        default = 0
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
device_id = parsed_args["device_id"]

device!(device_id)

dev = gpu

##=====

# im_array = load(datadir("exp_pro/Lsystems_1"))["im_array"]
# im_array = load(datadir("exp_pro/Lsystem_array_3.jld2"))["img_array"]
im_array = load(datadir("exp_pro/Lsystem_array_2lvl_thicker_2.jld2"))["img_array"]

frac_train = 0.9
n_train = trunc(Int, frac_train * size(im_array, 3))

train_ims = im_array[:, :, 1:n_train]
test_ims = im_array[:, :, n_train+1:end]

train_loader = DataLoader((dev(train_ims)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_ims)); batchsize=args[:bsz], shuffle=true,
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
## ===========


function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        return [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(
        HyDense(args[:π], 64, Θ[3], elu),
        flatten,
        HyDense(64, args[:imzprod], Θ[4], relu),
        flatten)

    z0 = fz.(Θ[5])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        return [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)
    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    return z1, a1, x̂, patch_t
end

function sample_(z, x; args=args)
    RNP_decoder(z, x, 3)
end

function stack_ims(xs; n=8)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end


function RNP_decoder(z, x, level; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    if level > 0
        out_ = RNP_decoder(z1, patch_t, level - 1)
        out = sample_patch(out_, a1, sampling_grid)
    else
        out = sample_patch(x̂, a1, sampling_grid)
    end
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        if level > 0
            out_ = RNP_decoder(z1, patch_t, level - 1)
            out += sample_patch(out_, a1, sampling_grid)
        else
            out += sample_patch(x̂, a1, sampling_grid)
        end
    end
    return out
end

## ==========
using MLUtils
args[:π] = 16
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:esz], args[:π])

mdec = Chain(
    HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:imzprod], args[:bsz], relu),
    flatten,
)

l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

l_enc_za_a = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:esz], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

modelpath = "saved_models/Lsystems/2lvl_hyper_try_thicker_reset/add_offset=true_asz=6_bsz=64_esz=32_ha_sz=32_img_channels=1_imzprod=2500_model_ind=0_scale_offset=2.0_seqlen=3_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.1_π=16_500eps.bson"

# modelpath = "saved_models/Lsystems/2lvl_hyper_try_thicker_reset/add_offset=true_asz=6_bsz=64_esz=32_ha_sz=32_img_channels=1_imzprod=2500_model_ind=0_scale_offset=2.0_seqlen=3_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.1_π=4_500eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## ======

args[:seqlen] = 3


x = sample_loader(test_loader)
let
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(x, inds)
    display(p)
end


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

begin
    args[:seqlen] = 3
    args[:scale_offset] = 2.0f0
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = sample_(z, x) |> cpu
    stack_ims(out[:, :, :, 1:9]; n=3) |> plot_digit
end

## =====

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


function rgb_to_orange(x)
    a = x
    b = 0.65f0 .* x
    c = 0.0f0 .* x
    cat(a, b, c, dims=3)
end

function rgb_to_orange(x)
    # a = x
    # b = 0.65f0 .* x
    # c = 0.0f0 .* x
    # cat(a, b, c, dims=3)
    x
end


function orange_on_rgb(xs)
    orange_patch = rgb_to_orange(xs[end])
    min.(cat(xs[1:3]..., dims=3) .+ orange_patch, 1.0f0)
end

function view_patches_rgb(patches, ind)
    im_array = orange_on_rgb(patches)
    colorview(RGB, permutedims(im_array[:, :, :, ind], [3, 1, 2]))
end

begin
    # z = randn(Float32, args[:π], args[:bsz]) |> gpu
    z = rand(Float32, args[:π], args[:bsz]) |> gpu
    recs, small_patches, patches, trans_patches, as, patches_t = sample_patches(z, x)
    a = [imresize(view_patches_rgb(trans_patches, ind), (100, 100)) for ind in 1:16]
    a = map(x -> collect(x'), a)
    stack_ims(cat(a..., dims=3); n=4)
end
