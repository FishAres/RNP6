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

im_array = load(datadir("exp_pro/Lsystems_1"))["im_array"]

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

function get_diag_bar(args)
    canv = zeros(Float32, args[:img_size]..., args[:bsz])
    for i in 5:24
        canv[i-4:i+4, i-4:i+4, :] .= 1.0f0
    end
    return canv
end

# curve_primitive = load("saved_models/two_primitive_comparison/curve_primitives/curve1.jld2")["curve_patch"]

## ==========

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

function sample_levels(z, x; args=args)
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

"overwriting to scale down offset to facilitate learning (not pushing things to edges)"
@inline function get_affine_mats(thetas; scale_offset=0.0f0)
    sc = (@view thetas[[1, 4], :]) .+ 1.0f0 .+ scale_offset
    # b = 0.75f0 .* sc .* (@view thetas[5:6, :]) # scaled by 0.75
    b = sc .* (@view thetas[5:6, :]) # scaled by 0.75
    A_rot = get_rot_mat(π32 * (@view thetas[2, :]))
    A_sc = unsqueeze(sc, 2) .* diag_mat
    A_shear = get_shear_mat(@view thetas[3, :])
    return A_rot, A_sc, A_shear, b
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    RLs, KLs = [], []
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)])
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                return log_value(lg, "KL loss", klqp)
            end
            Zygote.ignore() do
                push!(KLs, klqp)
                push!(RLs, rec_loss)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses, RLs, KLs
end


## ==========

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

Hx = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(LayerNorm(args[:π]),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz]; bias=false)
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
        flatten)
    outsz = Flux.outputsize(enc1,
        (args[:img_size]..., args[:img_channels],
            args[:bsz]))
    Chain(enc1,
        Dense(outsz[1], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Split(Dense(64, args[:π]), Dense(64, args[:π])))
end)

nps = sum([prod(size(p)) for p in Flux.params(Hx, Ha)])
ps = Flux.params(Hx, Ha, Encoder)
## =====

x = first(test_loader)
let
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(x, inds)
    display(p)
end

## =====
save_folder = "Lsystems"
alias = "2lvl_hyper_try0"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 3
args[:scale_offset] = 1.8f0
args[:λ] = 1.0f-6
args[:λpatch] = 0.0f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.1f0

args[:η] = 1e-4
args[:model_ind] = Symbol(parsed_args["model_ind"])

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
log_value(lg, "lambda_patch", args[:λpatch])
## ====
begin
    Ls, RLs, KLs, TLs = [], [], [], []
    for epoch in 1:500
        if epoch % 25 == 0
            opt.eta = 0.8 * opt.eta
            log_value(lg, "learning_rate", opt.eta)
        end

        ls, rec_losses, klqps = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)

        if epoch % 5 == 0
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = sample_(z, x)
            out, out_x̂ = sample_levels(z, x)
            psamp_out = plot_digit(stack_ims(out), title="2 level")
            psamp_x̂ = plot_digit(stack_ims(out_x̂), title="1 level")
            psamp = plot(psamp_out, psamp_x̂)
            log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)
        end

        Lval = test_model(test_loader)
        log_value(lg, "test loss", Lval)
        @info "Test loss: $Lval"
        if epoch % 50 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
        push!(Ls, ls)
        push!(RLs, rec_losses)
        push!(KLs, klqps)
        push!(TLs, Lval)
    end
end