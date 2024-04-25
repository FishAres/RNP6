using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader

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
    :layer_sz => 64, # layer size for primary networks
    :f_z => identity, # state RNN activation
    :α => 1.0f0, # rec. loss weighting
    :β => 0.1f0, # KL weighting
    :add_offset => true, # add scale offset to affine transforms
    :asz => 6, # of spatial transformer params
    :seqlen => 3, # RNP sequence length
    :lr => 1e-4, # learning rate
    :λ => 1.0f-4, # hypernet regularization weight
    :λpatch => 0, # patch reconstruction regularization weight
    :scale_offset => 1.6f0, # affine transform scale offset
)
args[:imzprod] = prod(args[:img_size])

## =====

# device!(0) # specify gpu if there are multiple
const dev = has_cuda() ? gpu : cpu

## ==== Load MNIST data

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]

train_loader = DataLoader((dev(train_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)

## =====
# constant matrices for "nice" affine transformation
const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))

const Hstate_bounds, Hpolicy_bounds = get_primary_bounds(args)

## ====
function save_model(model, savestring)
    model_state = Flux.state(model |> cpu)
    jldsave(savestring * ".jld2"; model_state)
    println("saved at $savestring")
end

function load_model(model, savestring)
    ms = JLD2.load(savestring * ".jld2", "model_state")
    Flux.loadmodel!(model, ms)
end
## ==== Get networks
model = H_state, H_policy, Encoder = generate_hypernets(args) |> dev
ps = Flux.params(H_state, H_policy, Encoder)
## ====

x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)

plot_recs(x, inds)

## =====
save_folder, alias = "RNP_2lvl_mnist", ""
save_dir = get_save_dir(save_folder, alias)

## ====
lg = new_logger(joinpath(save_folder, alias), args)

losses = train_model(ps, train_loader, test_loader, save_dir; n_epochs=50, save_every=50, logger=lg, clip_grads=true)

## =====

# function sample_patches(z, x; args=args)
#     outputs = recs, comb_patches, patches, trans_patches, as, patches_t = [], [], [], [], [], [], []
#     θsz = H_state(z)
#     θsa = H_policy(z)
#     models, z0, a0 = get_models(θsz, θsa; args=args)
#     z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
#     small_patches, out_small = full_sequence_patches(z1, patch_t)
#     out = sample_patch(out_small, a1, sampling_grid)
#     push_to_arrays!((out, small_patches, out_small, out, a1, patch_t), outputs)
#     for t in 2:args[:seqlen]
#         z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
#             scale_offset=args[:scale_offset])
#         small_patches, out_small = full_sequence_patches(z1, patch_t)
#         trans_patch = sample_patch(out_small, a1, sampling_grid)
#         out += trans_patch
#         push_to_arrays!((out, small_patches, out_small, trans_patch, a1, patch_t), outputs)
#     end
#     return outputs
# end


# include(srcdir("model_utils.jl"))

# function view_patches_rgb2(patches, ind)
#     im_array = orange_on_rgb(patches)
#     colorview(RGB, permutedims(im_array[:, :, :, ind], [3, 1, 2]))
# end


# begin
#     z = randn(Float32, args[:π], args[:bsz]) |> gpu
#     recs, small_patches, patches, trans_patches, as, patches_t = sample_patches(z, x)
#     a = [imresize(view_patches_rgb2(trans_patches, ind), (60, 60)) for ind in 1:16]
#     a = map(x -> collect(x'), a)
#     stack_ims(cat(a..., dims=3); n=4)
# end

# function push_arrays_to_dict!(dict::Dict, data::Tuple)
#     [push!(dict[key], cpu(val)) for (key, val) in zip(keys, data)]
# end

# function full_sequence_patches(models::Tuple, z0, a0, x; args=args,
#     scale_offset=args[:scale_offset])
#     patches, trans_patches = [], []
#     f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
#     z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
#     trans_patch = sample_patch(x̂, a1, sampling_grid)
#     out = trans_patch
#     push!(patches, cpu(out))
#     push!(trans_patches, cpu(trans_patch))
#     for t in 2:args[:seqlen]
#         z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
#         trans_patch = sample_patch(x̂, a1, sampling_grid)
#         out += trans_patch
#         push!(patches, cpu(out))
#         push!(trans_patches, cpu(trans_patch))
#     end
#     return patches, trans_patches, out
# end

# function full_sequence_patches(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
#     θsz = H_state(z)
#     θsa = H_policy(z)
#     models, z0, a0 = get_models(θsz, θsa; args=args)
#     return full_sequence_patches(models, z0, a0, x; args=args, scale_offset=scale_offset)
# end

# function get_loop_patches(x; args=args)
#     # i like symbols
#     keys = [:recs, :comb_patches, :trans_patches, :patches, :as, :patches_t]
#     output_dict = Dict(key => [] for key in keys)
#     r = gpu(randn(args[:π], args[:bsz]))
#     μ, logvar = Encoder(x)
#     z = sample_z(μ, logvar, r)
#     θsz = H_state(z)
#     θsa = H_policy(z)
#     models, z0, a0 = get_models(θsz, θsa; args=args)
#     z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
#     small_patches, trans_patches, out_small = full_sequence_patches(z1, patch_t)
#     recs = sample_patch(out_small, a1, sampling_grid)
#     push_arrays_to_dict!(output_dict, (recs, small_patches, trans_patches, out_small, a1, patch_t))
#     for t in 2:args[:seqlen]
#         z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
#             scale_offset=args[:scale_offset])
#         small_patches, trans_patches, out_small = full_sequence_patches(z1, patch_t)
#         trans_patch = sample_patch(out_small, a1, sampling_grid)
#         recs += trans_patch
#         push_arrays_to_dict!(output_dict, (recs, small_patches, trans_patches, out_small, a1, patch_t))
#     end
#     return output_dict
# end

# outputs = get_loop_patches(x)
