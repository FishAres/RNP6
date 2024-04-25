using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64,
    :img_size => (28, 28),
    :π => 8, # latent dim size
    :img_channels => 1,
    :esz => 32, # RNN input size
    :ha_sz => 32, # policy RNN hidden size 
    :layer_sz => 64,
    :add_offset => true,
    :f_z => identity,
    :asz => 6, # of spatial transformer params
    :seqlen => 3,
    :scale_offset => 1.6f0,
)
args[:imzprod] = prod(args[:img_size])

## =====

dev = has_cuda() ? gpu : cpu

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

const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))

## ====
const Hstate_bounds, Hpolicy_bounds = get_primary_bounds(args)
## ====
modelpath = "saved_models/RNP_2lvl_mnist/add_offset=true_asz=6_bsz=64_esz=32_ha_sz=32_img_channels=1_imzprod=784_layer_sz=64_lr=0.0001_scale_offset=1.6_seqlen=3_α=1.0_β=0.1_λ=0.0001_λpatch=0_π=8_50eps"

model = H_state, H_policy, Encoder = generate_hypernets(args) |> dev

load_model(model, modelpath)

## === TSne
"""Extend `get_loop` to return z2 and z1s
#Arguments
- `x::Array`: Input image
- `use_mean::Bool`: Use encoded mean (true) for z² or reparameterize (false)
"""
function get_loop_zs(x; args=args, use_mean=true)
    output_list = z1s, recs, patches, as, patches_t = [[] for _ in 1:5]
    μ, logvar = Encoder(x)
    z = use_mean ? μ : sample_z(μ, logvar, dev(randn(Float32, args[:π], args[:bsz])))
    θs_state = H_state(z)
    θs_policy = H_policy(z)
    models, z0, a0 = get_models(θs_state, θs_policy; args=args)
    z_t, a_t, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z_t, patch_t)
    output = sample_patch(out_small, a_t, sampling_grid)
    push_to_arrays!((z_t, output, out_small, a_t, patch_t), output_list)
    for t in 2:args[:seqlen]
        z_t, a_t, x̂, patch_t = forward_pass(z_t, a_t, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z_t, patch_t)
        output += sample_patch(out_small, a_t, sampling_grid)
        push_to_arrays!((z_t, output, out_small, a_t, patch_t), output_list)
    end
    return μ |> cpu, output_list
end

"Stack it!"
function get_stacked_zs_ims(test_loader; use_mean=true)
    z2s, z1s, outputs, patch_outputs = [[] for _ in 1:4]
    for (i, x) in enumerate(test_loader)
        z2, (z1_, recs, patches, as, patches_t) = get_loop_zs(x; use_mean=use_mean)
        push!(z2s, z2)
        push!(z1s, hcat(z1_...))
        push!(outputs, recs[end])
        push!(patch_outputs, cat(patches..., dims=4))
    end
    z2s = reduce(hcat, z2s)
    z1s = reduce(hcat, z1s)
    outputs = cat(outputs..., dims=4)
    patch_outputs = cat(patch_outputs..., dims=4)
    return z2s, z1s, outputs, patch_outputs
end

"Get `n_samples` random samples from the z² and z¹ vectors"
function get_tsne_samples(z2s, z1s, outputs, patch_outputs, n_samples)
    ims = cat(outputs, patch_outputs, dims=4)
    zs = hcat(z2s, z1s)
    sample_inds = sample(1:size(zs, 2), n_samples, replace=false)
    z_tsne = zs[:, sample_inds]
    ims_tsne = ims[:, :, 1, sample_inds]
    return z_tsne, ims_tsne
end

using TSne, Clustering

function get_nearest_point(Ys, R::KmeansResult, cluster_ind)
    centroid = argmin(sqrt.((Ys[:, 1] .- R.centers[1, cluster_ind]) .^ 2) + sqrt.((Ys[:, 2] .- R.centers[2, cluster_ind]) .^ 2))
    return centroid
end


function plot_tsne_clusters_ims(Ys, cluster_ims, R::KmeansResult)
    inds = assignments(R)
    n_clusters = length(unique(inds))
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
    return p
end

function plot_tsne_clusters_ims(Ys, n_clusters::Int)
    R = kmeans(Ys', n_clusters)
    cluster_ims = [ims_tsne[:, :, get_nearest_point(Ys, R, k)] for k in 1:n_clusters]
    plot_tsne_clusters_ims(Ys, cluster_ims, R)
end

## === Run it!
z2s, z1s, outputs, patch_outputs = get_stacked_zs_ims(test_loader)
Ys = tsne(zs[:, sample_inds]') # perform t-SNE

n_clusters = 40 # this is a decent setting
plot_tsne_clusters_ims(Ys, n_clusters)


