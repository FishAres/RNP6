using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader

include(srcdir("eth80_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, # batch size
    :img_size => (50, 50), # note larger image size for ETH80
    :π => 16, # latent dim size / state RNN hidden dim size
    :img_channels => 3,
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

#device!(0)  # specify gpu if there are multiple
const dev = has_cuda() ? gpu : cpu

## =====#
# Run src/process_eth80.jl to get the processed ETH80 data
eth80_train = load(datadir("exp_pro", "eth80_segmented_train.jld2"))["eth80_train"]
eth80_test = load(datadir("exp_pro", "eth80_segmented_test.jld2"))["eth80_test"]

train_loader = DataLoader((dev(eth80_train)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(eth80_test)); batchsize=args[:bsz], shuffle=true,
    partial=false)

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

## =====

H_state, H_policy, Encoder = generate_hypernets(args) |> dev
ps = Flux.params(H_state, H_policy, Encoder)
## =====


x = first(test_loader)
inds = sample(1:args[:bsz], 6; replace=false)
let
    p = plot_recs(x, inds)
    display(p)
end

## =====
save_folder, alias = "RNP_2lvl_eth80", ""
save_dir = get_save_dir(save_folder, alias)

## =====
opt = ADAM(args[:lr])
lg = new_logger(joinpath(save_folder, alias), args)


begin
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = sample_(z, x)[:, :, :, 1:16]
    psamp = stack_ims(out, n=4)
    stack_ims(out, n=4) |> imview_color |> x -> imresize(x, (400, 400))
end

