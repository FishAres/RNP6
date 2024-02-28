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

## ======
args[:π] = 16
# modelpath = "saved_models/hypernet_2lvl/mnist_2lvl/a_sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=1_imzprod=784_scale_offset=2.4_seqlen=4_α=1.0_β=0.1_η=0.0001_λ=1e-6_λpatch=0.0_π=16_200eps.bson"

# modelpath = "saved_models/hypernet_2lvl/mnist_2lvl_inc_λpatch/a_sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=1_imzprod=784_scale_offset=2.4_seqlen=4_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.04_π=16_100eps.bson"

modelpath = "saved_models/hypernet_2lvl/mnist_2lvl_inc_λpatch/a_sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=1_imzprod=784_scale_offset=2.4_seqlen=4_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.08_π=16_200eps.bson"


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

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

model_args_dict = parse_savename(
    "sample_len=8_add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_ha_sz=32_img_channels=1_imzprod=784_scale_offset=2.4_seqlen=4_α=1.0_β=0.5_η=0.0001_λ=1e-6_λpatch=0.08_π=16"
)

args = update_args_dict(model_args_dict, args)

args[:seqlen] = 4
args[:scale_offset] = 2.4f0
## ======
x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)
let
    p = plot_recs(x, inds)
    display(p)
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

function sample_levels(z, x; args=args)
    x = first(test_loader)
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

function get_nearest_point(Ys, R::KmeansResult, cluster_ind)
    centroid = argmin(sqrt.((Ys[:, 1] .- R.centers[1, cluster_ind]) .^ 2) + sqrt.((Ys[:, 2] .- R.centers[2, cluster_ind]) .^ 2))
    return centroid
end

function get_centroid_image(Ys, R, Z, cluster_ind; plot=false)
    x = first(test_loader)
    zi = get_nearest_point(Ys, R, cluster_ind)
    Zi = repeat(Z[:, zi], 1, args[:bsz]) |> gpu
    out, out_x̂ = sample_levels(Zi, x) |> cpu
    if plot
        return plot_digit(out_x̂[:, :, 1, 1])
    else
        return out_x̂[:, :, 1, 1]
    end
end

## =====
using Clustering, TSne
using Flux, Zygote

Z = let
    z = []
    for x in test_loader
        z2, z1s = get_μ_zs(x)
        push!(z, hcat(z2, hcat(z1s...)))
    end
    hcat(z...)
end

Z_ = hcat(shuffle(collect(eachcol(Z)))...)[:, 1:5120]
Y = tsne(Z_')

Y

n_clusters = 40
R = kmeans(Y', n_clusters)
inds = assignments(R)

function get_cluster_coords(cluster_ind)
    zi = get_nearest_point(Y, R, 1)
    Z_[:, zi]
end
get_centroid_image(Y, R, Z_, 1)

ind = 0
begin
    ind += 1
    # z = Y[:, inds.==ind][:, 1:args[:bsz]] |> gpu
    # _, out = sample_levels(z, x)
    # stack_ims(out) |> plot_digit
    get_centroid_image(Y, R, Z_, ind) |> plot_digit
    title!("$ind")

end


begin
    # 3 -> 5
    opt = Descent(0.12)
    y1 = R.centers[:, 3]
    y2 = R.centers[:, 9]
    Is = []
    ps = Flux.params(y1)
    for i in 1:50
        loss, grad = withgradient(ps) do
            sum(y2 .- y1) .^ 2
        end

        Flux.update!(opt, ps, grad)

        z_ = repeat(y1, 1, 64) |> gpu
        _, out = sample_levels(z_, x)
        out = out[:, :, 1, 1]
        push!(Is, out)

    end
    cat(Is..., dims=3) |> stack_ims |> plot_digit
end

