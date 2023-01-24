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

all_chars = load("../Recur_generative/data/exp_pro/omniglot_train.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
num_train = trunc(Int, 0.9 * length(xs))

new_chars = load("../Recur_generative/data/exp_pro/omniglot_eval.jld2")
xs_new = shuffle(vcat((new_chars[key] for key in keys(new_chars))...))

function fast_cat(xs)
    x_array = zeros(Float32, size(xs[1])..., length(xs))
    Threads.@threads for i in 1:length(xs)
        x_array[:, :, i] = xs[i]
    end
    return x_array
end

xs_cat = fast_cat(xs)
train_chars = xs_cat[:, :, 1:num_train]
val_chars = xs_cat[:, :, (num_train+1):end]

new_xs_cat = fast_cat(xs_new)
## ==== KMnist

kmnist_ = load(datadir("exp_pro", "kmnist.jld2"))["ims"]
kmnist_ = shuffle(collect(eachslice(permutedims(kmnist_, [2, 3, 1]), dims=3)))

kmnist_train = kmnist_[1:54000]
kmnist_test = kmnist_[54001:end]
train_data = cat(fast_cat(kmnist_train), train_chars, dims=3)
test_data = cat(fast_cat(kmnist_test), val_chars, dims=3)

## =====
train_loader = DataLoader(dev(train_data); batchsize=args[:bsz], shuffle=true,
    partial=false)
val_loader = DataLoader(dev(test_data); batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(dev(new_xs_cat); batchsize=args[:bsz], shuffle=true, partial=false)

## =====

dev = has_cuda() ? gpu : cpu

const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))

## ===== functions

function get_param_sizes(model)
    nw = []
    for m in Flux.modules(model)
        if hasproperty(m, :weight)
            wprod = prod(size(m.weight)[1:(end-1)])
            if hasproperty(m, :bias)
                wprod += size(m.bias)[1]
            end
            push!(nw, wprod)
        end
    end
    return nw
end

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        return [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=elu)

    # f_state => μ, logσ
    state_to_z = Chain(HyDense(args[:π], 2 * args[:π], Θ[3], identity), flatten)

    # conv decoder
    Dec_z_x̂ = Chain(HyDense(args[:π], 64, Θ[4], elu),
        flatten,
        HyDense(64, 64, Θ[5], elu),
        flatten,
        HyDense(64, 64, Θ[6], elu),
        flatten,
        HyDense(64, 64, Θ[7], elu),
        flatten,
        HyDense(64, args[:imzprod], Θ[8], relu),
        flatten)
    z0 = fz.(Θ[9])

    # return (Enc_za_z, f_state, state_to_z, Dec_z_x̂), z0
    return (f_state, Enc_za_z, state_to_z, Dec_z_x̂), z0
end


function get_models(θsz; args=args, Hx_bounds=Hx_bounds)
    (f_state, Enc_za_z, state_to_z, Dec_z_x̂), z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    models = f_state, Enc_za_z, state_to_z, Dec_z_x̂
    return models, z0
end

function get_models(z; args=args)
    θsz = Hx(z)
    models, z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    return models, z0
end

Zygote.@nograd function get_a1s(args)
    a1 = rand(Uniform(-0.75f0, 0.75f0), 6, args[:bsz])
    a1 = sin.(a1) |> gpu
end

"one iteration"
function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, Enc_za_z, state_to_z, Dec_z_x̂ = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    z1_ = f_state(ez)
    z1_mv = state_to_z(z1_)
    μ1, logvar1 = z1_mv[1:args[:π], :], z1_mv[args[:π]+1:end, :]
    z1 = sample_z(μ1, logvar1, randn(Float32, args[:π], args[:bsz]) |> gpu)

    x̂ = Dec_z_x̂(z1)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    kl1 = kl_loss(μ1, logvar1)

    return z1, x̂, patch_t, kl1
end

function model_loss(x; args=args)
    kl1_frac = Float32(1 / args[:seqlen])
    a1s = [get_a1s(args) for _ in 1:args[:seqlen]]
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, randn(Float32, args[:π], args[:bsz]) |> gpu)
    θsz = Hx(z)

    models, z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    f_state, Enc_za_z, state_to_z, Dec_z_x̂ = models
    z1, x̂, patch_t, kl1 = forward_pass(z0, a1s[1], models, x;
        scale_offset=args[:scale_offset])
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t in 2:args[:seqlen]
        z1, x̂, patch_t, kl1_ = forward_pass(z1, a1s[t], models, x;
            scale_offset=args[:scale_offset])

        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
        kl1 += kl1_
    end
    klqp = kl_loss(μ, logvar)
    return Lpatch, klqp + kl1_frac * kl1
end

function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    r = gpu(rand(args[:D], args[:π], args[:bsz]))

    a1s = [get_a1s(args) for _ in 1:args[:seqlen]]
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    models, z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    f_state, Enc_za_z, state_to_z, Dec_z_x̂ = models
    z1, x̂, patch_t, kl1 = forward_pass(z0, a1s[1], models, x;
        scale_offset=args[:scale_offset])
    out_small = sample_patch(x̂, a1s[1], sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    push_to_arrays!((out_small, x̂, a1s[1], patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, x̂, patch_t, kl1_ = forward_pass(z1, a1s[t], models, x;
            scale_offset=args[:scale_offset])
        out_small += sample_patch(x̂, a1s[t], sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
        kl1 += kl1_
        push_to_arrays!((out_small, x̂, a1s[t], patch_t), outputs)
    end
    return outputs
end

function get_x_patches(x; args=args)
    x_ = dropdims(zoom_in2d(x, get_a1s(args), sampling_grid), dims=3)
    x_ = collect(eachslice(x_, dims=3))
    x__ = collect(eachslice(x, dims=3))
    x_cat = shuffle([x_; x__])
    x_cat = sample(x_cat, args[:bsz], replace=false)
    cat(x_cat..., dims=3)
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    for (i, x) in enumerate(train_data)
        x = get_x_patches(x)

        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * norm(Flux.params(Hx))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        x = get_x_patches(x)
        rec_loss, klqp = model_loss(x)
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

function plot_recs(x, inds; plot_seq=true, args=args)
    x = get_x_patches(x)
    full_recs, patches, xys, patches_t = get_loop(x)
    p = if plot_seq
        let
            patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
            [plot_rec(full_recs[end], x, patches_, ind) for ind in inds]
        end
    else
        [plot_rec(patches[end], x, ind) for ind in inds]
    end

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

## ====== model

args[:π] = 96
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:esz], args[:π]) # μ, logvar
l_fx_z = (args[:π] * 2 * args[:π])

mdec = Dec_z_x̂ = Chain(HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:imzprod], args[:bsz], relu))

l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_fx_z; l_dec_x...]

Encoder = let
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
        Dense(64, 64), LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Split(Dense(64, args[:π]), Dense(64, args[:π])))
end |> gpu

Hx = Chain(
    LayerNorm(args[:π], 64),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

ps = Flux.params(Hx, Encoder)

## =====
let
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end

## =====

save_folder = "forward_first"
alias = "mnist_hybrid_digits_patches_v0"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 6
args[:scale_offset] = 2.4f0
args[:λ] = 1.0f-4
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.1f0
args[:η] = 1e-4

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)

## ====

begin
    Ls = []
    for epoch in 1:250
        if epoch % 50 == 0
            opt.eta = 0.8 * opt.eta
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6; replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)

        Ltest = test_model(test_loader)
        log_value(lg, "test_loss", Ltest)
        @info "Test loss: $Ltest"
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((Hx, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end


