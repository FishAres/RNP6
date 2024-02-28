using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using Flux.Data: DataLoader
include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, # batch size
    :img_size => (28, 28),
    :π => 32, # latent dim size / state RNN hidden dim size
    :img_channels => 1,
    :esz => 32, # RNN input size
    :ha_sz => 32, # policy RNN hidden size
    :layer_sz => 64, # layer size for primary networks
    :f_z => elu, # state RNN activation
    :α => 1.0f0, # rec. loss weighting
    :β => 0.1f0, # KL weighting
    :add_offset => true, # add scale offset to affine transforms
    :asz => 6, # of spatial transformer params
    :seqlen => 4, # RNP sequence length
    :lr => 1e-4, # learning rate
    :λ => 1.0f-4, # hypernet regularization weight
    :λpatch => 0, # patch reconstruction regularization weight
    :scale_offset => 2.0f0, # affine transform scale offset
)
args[:imzprod] = prod(args[:img_size])

## =====

#device!(0)
const dev = has_cuda() ? gpu : cpu

##=====

function process_chars(digits; imsize=(28, 28))
    resized = imresize(digits, imsize...)
    background_sub = 1.0f0 .- resized
    return background_sub
end

function process_omni_data(frac_train; args=args)
    train_chars, _ = Omniglot(; split=:train)[:]
    test_chars, _ = Omniglot(; split=:test)[:]
    train_chars = process_chars(train_chars, imsize=args[:img_size])
    test_set = process_chars(test_chars, imsize=args[:img_size])
    n_train = trunc(Int, frac_train * size(train_chars)[end])
    train_set = train_chars[:, :, 1:n_train]
    val_set = train_chars[:, :, n_train+1:end]
    return train_set, val_set, test_set
end

train_set, val_set, test_set = process_omni_data(0.85)

train_loader = DataLoader(dev(train_set); batchsize=args[:bsz], shuffle=true,
    partial=false)
val_loader = DataLoader(dev(val_set); batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(dev(test_set); batchsize=args[:bsz], shuffle=true, partial=false)


## =====
# constant matrices for "nice" affine transformation
const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))

## ====

const Hstate_bounds, Hpolicy_bounds = get_primary_bounds(args)

H_state, H_policy, Encoder = generate_hypernets(args) |> dev

ps = Flux.params(H_state, H_policy, Encoder)

## ====
x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)
let
    p = plot_recs(x, inds)
    display(p)
end

## =====
save_folder, alias = "RNP_2lvl_omni", ""
save_dir = get_save_dir(save_folder, alias)

lg = new_logger(joinpath(save_folder, alias), args)
losses = train_model(ps, train_loader, test_loader, save_dir; n_epochs=5, logger=lg)

println("Done!")

## =====

function get_μ_zs(x)
    z1s = []
    μ, logvar = Encoder(x)
    θsz = H_state(μ)
    θsa = H_policy(μ)
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

using StatsBase: shuffle
function shuffle_batchwise(x1, x2)
    b = shuffle([x1; x2])
    return hcat(b...)
end

function sample_levels(z, x; args=args)
    θsz = H_state(z)
    θsa = H_policy(z)
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

n_clusters = 10
R = kmeans(Ys', n_clusters)
inds = assignments(R)

begin
    p = plot(legend=false)
    for i in 1:n_clusters
        scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
    end
    p
end
using Flux: batch
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