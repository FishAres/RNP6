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
