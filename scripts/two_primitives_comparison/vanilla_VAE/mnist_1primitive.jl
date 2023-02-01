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
using Images: imresize
include(srcdir("vanilla_VAE_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, # batch size
    :img_size => (28, 28),
    :π => 16, # latent dim size
    :img_channels => 1, # number of image channels
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

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((dev(train_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)

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
    bar_resized = imresize(bar, (6, 6))[:, :, 1]
    bar_conv = repeat(bar_resized, 1, 1, 1, 1)
end

const dec_filters = line_primitive |> gpu
## -====
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


const dec_cdims = let
    h = randn(12, 12, 1, args[:bsz]) |> gpu
    m = ConvTranspose((6, 6), 1 => 1, stride=2)
    cdims = Flux.conv_transpose_dims(gpu(m), h)
    cdims
end

Decoder = Chain(
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 256),
    LayerNorm(256, elu),
    x -> reshape(x, 8, 8, 4, :),
    ConvTranspose((4, 4), 4 => 64, stride=1, pad=0),
    BatchNorm(64, elu),
    BasicBlock(64 => 64, +),
    BasicBlock(64 => 64, +),
    BasicBlock(64 => 64, +),
    ConvTranspose((4, 4), 64 => 1, stride=1, pad=1),
    BatchNorm(1, elu),
    x -> relu.(∇conv_data(x, dec_filters, dec_cdims))
) |> gpu

nps = sum([prod(size(p)) for p in Flux.params(Decoder)])

ps = Flux.params(Encoder, Decoder)
## =====
let
    x = first(test_loader)
    inds = sample(1:args[:bsz], 6; replace=false)
    p = plot_recs(x, inds)
    display(p)
end

## =====
save_folder = "one_primitive_comparison"
alias = "mnist_vanilla_VAE"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:λ] = 1.0f-6
args[:α] = 1.0f0
args[:β] = 0.1f0
args[:model_ind] = Symbol(parsed_args["model_ind"])
args[:η] = 1e-4

opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)

## ====
begin
    RLs, Ls, KLs, TLs = [], [], [], []
    for epoch in 1:200
        if epoch % 25 == 0
            opt.eta = 0.67 * opt.eta
            log_value(lg, "learning_rate", opt.eta)
        end
        ls, klqps, rec_losses = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)

        if epoch % 5 == 0
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = sample_(z)
            psamp = plot_digit(stack_ims(out))
            log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)
        end

        Lval = test_model(test_loader)
        log_value(lg, "test loss", Lval)
        @info "Test loss: $Lval"
        if epoch % 50 == 0
            save_model((Decoder, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
        push!(Ls, ls)
        push!(RLs, rec_losses)
        push!(KLs, klqps)
        push!(TLs, Lval)
    end
end

## =====

loss_dict = Dict("losses" => vcat(Ls...), "KLs" => vcat(KLs...), "RLs" => vcat(RLs...), "Test_losses" => vcat(TLs...), "n_dec_params" => nps)

JLD2.@save joinpath(save_dir, "$(savename(args))_$(args[:model_ind]).jld2") loss_dict
println("saved loss curves $(savename(args))")


