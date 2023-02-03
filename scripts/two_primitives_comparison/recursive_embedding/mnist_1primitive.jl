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
    :π => 32,
    :img_channels => 1,
    :esz => 32,
    :add_offset => true,
    :fa_out => sin,
    :f_z => elu,
    :asz => 6,
    :ha_sz => 32,
    :glimpse_len => 4,
    :seqlen => 5,
    :λ => 1.0f-3,
    :λpatch => Float32(1 / 4),
    :scale_offset => 2.8f0,
    :D => Normal(0.0f0, 1.0f0))
args[:imzprod] = prod(args[:img_size])
## =====

"parse command line arguments"
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
        "--program_size"
        arg_type = Int
        default = 16
        "--RNN_type"
        arg_type = String
        default = "GRU"
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

device_id = parsed_args["device_id"]
device!(device_id)

println(parsed_args["model_ind"])

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

const sampling_grid = (dev(get_sampling_grid(args[:img_size]...)))[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = dev(ones(1, 1, args[:bsz]))
const zeros_vec = dev(zeros(1, 1, args[:bsz]))
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = dev(cat(diag_vec...; dims=3))
const diag_off = dev(cat(1.0f-6 .* diag_vec...; dims=3))
## =====

function get_diag_bar(args)
    canv = zeros(Float32, 28, 28, args[:bsz])
    for i in 5:24
        canv[i-4:i+4, i-4:i+4, :] .= 1.0f0
    end
    return canv
end

line_primitive = let
    bar = get_diag_bar(args)
    flatten(bar)[:, 1:1]
end

const dec_filters = line_primitive |> gpu

## +====
function forward_pass1(z2, z1, a1, x; scale_offset=args[:scale_offset])
    za = vcat(z1, a1)
    ez = Enc_za_z1(za)
    ea = Enc_za_a1(za)
    z1 = f_state1(vcat(z2, ez))
    a1 = Dec_z_a1(f_policy1(vcat(z2, ea)))

    x̂ = dec_z_x̂(z1, Dec_z_x̂1, dec_filters)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    return z1, a1, x̂, patch_t
end

function dec_z_x̂(z, Dec_z_x̂, dec_filters)
    h = Dec_z_x̂(z)
    relu.(dec_filters * h)
end

function full_sequence0(z2, x; args=args,
    scale_offset=args[:scale_offset])
    # Flux.reset!(f_state1)
    # Flux.reset!(f_policy1)
    z0 = z0_net(z2)
    a0 = a0_net(z2)
    z1, a1, x̂, patch_t = forward_pass1(z2, z0, a0, x; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass1(z2, z1, a1, x; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end


function model_loss(x, r; args=args)
    # [Flux.reset!(m) for m in (f_state1, f_policy1)]
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    z0 = z0_net(z)
    a0 = a0_net(z)
    z1, a1, x̂, patch_t = forward_pass1(z, z0, a0, x; scale_offset=args[:scale_offset])
    out_small = full_sequence0(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass1(z, z1, a1, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence0(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

"output sequence: full recs, local recs, xys (a1), patches_t"
function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    # [Flux.reset!(m) for m in (f_state1, f_policy1)]
    r = gpu(rand(args[:D], args[:π], args[:bsz]))
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    z0 = z0_net(z)
    a0 = a0_net(z)
    z1, a1, x̂, patch_t = forward_pass1(z, z0, a0, x; scale_offset=args[:scale_offset])
    out_small = full_sequence0(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass1(z, z1, a1, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence0(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end

function tile_batch(out; n=8)
    a = collect(eachslice(dropdims(out; dims=3); dims=3))
    b = collect(partition(a, n))
    c = map(x -> hcat(x...), b[1:n])
    return vcat(c...)
end

function random_sample(z, x; args=args)
    [Flux.reset!(m) for m in (f_state1, f_policy1)]
    z0 = z0_net(z)
    a0 = a0_net(z)
    z1, a1, x̂, patch_t = forward_pass1(z, z0, a0, x; scale_offset=args[:scale_offset])
    out_small = full_sequence0(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass1(z, z1, a1, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence0(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return out |> cpu
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)])
    rec_losses, kls = [], []
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
                push!(rec_losses, rec_loss)
                push!(kls, klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses, rec_losses, kls
end

function test_model(test_data; D=args[:D])
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)])
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

## +=====
args[:π] = parsed_args["program_size"]

RNN_type_string = parsed_args["RNN_type"]
args[:RNN_type] = RNN_type_string

RNN_type = RNN_type_string == "GRU" ? GRU : LSTM


Enc_za_z1 = Chain(Dense(args[:π] + args[:asz], args[:esz]), LayerNorm(args[:esz], elu)) |> gpu

Enc_za_a1 = Chain(Dense(args[:π] + args[:asz], args[:esz],), LayerNorm(args[:esz], elu)) |> gpu

f_state1 = Chain(RNN_type(args[:esz] + args[:π], args[:π],), LayerNorm(args[:π], elu)) |> gpu

f_policy1 = Chain(RNN_type(args[:esz] + args[:π], args[:ha_sz],), LayerNorm(args[:ha_sz], elu)) |> gpu

Dec_z_x̂1 = Chain(
    Dense(args[:π], 64, elu),
    Dense(64, 1, elu),
) |> gpu

Dec_z_a1 = Dense(args[:ha_sz], args[:asz], sin) |> gpu

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
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Split(Dense(64, args[:π]), Dense(64, args[:π])))
end |> gpu

z0_net = Dense(args[:π], args[:π], elu) |> gpu
a0_net = Dense(args[:π], args[:asz], elu) |> gpu

ps = Flux.params(Encoder, Enc_za_z1, Enc_za_a1,
    f_state1, f_policy1, Dec_z_x̂1, Dec_z_a1, z0_net, a0_net)

model = (Encoder, Enc_za_z1, Enc_za_a1,
    f_state1, f_policy1, Dec_z_x̂1, Dec_z_a1, z0_net, a0_net)


nps = sum([prod(size(p)) for p in Flux.params(Enc_za_z1, Enc_za_a1,
    f_state1, f_policy1, Dec_z_x̂1, Dec_z_a1, z0_net, a0_net)])

println("Number of decoder params: $nps")
## =====

let
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end
## =====

save_folder = "one_primitive_comparison"
alias = "mnist_2lvl_embedding_RNN"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 4
args[:scale_offset] = 2.2f0

args[:λpatch] = 0.0f0
args[:λ] = 1.0f-6
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.1f0

args[:η] = 1e-4
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls, RLs, KLs, TLs = [], [], [], []
    x = sample_loader(test_loader)
    for epoch in 1:200
        if epoch % 10 == 0
            opt.eta = 0.8 * opt.eta
        end
        ls, rec_losses, kls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6; replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        psamp = let
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = random_sample(z, x)
            plot_digit(tile_batch(out))
        end
        display(psamp)
        log_image(lg, "random_samples_$(epoch)", psamp)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        push!(KLs, kls)
        push!(RLs, rec_losses)
        push!(TLs, L)
        if epoch % 50 == 0
            save_model(model, joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

##
Ls, KLs, RLs, TLs = [], [], [], []
nps = sum([prod(size(p)) for p in Flux.params(Enc_za_z1, Enc_za_a1,
    f_state1, f_policy1, Dec_z_x̂1, Dec_z_a1, z0_net, a0_net)])

args[:model_ind] = parsed_args["model_ind"]
loss_dict = Dict("losses" => vcat(Ls...), "KLs" => vcat(KLs...), "RLs" => vcat(RLs...), "Test_losses" => vcat(TLs...), "n_dec_params" => nps)

JLD2.@save joinpath(save_dir, "$(savename(args))_$(args[:model_ind]).jld2") loss_dict
println("saved loss curves $savename(args)")

