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
using Images
using Plots

include(srcdir("hypernet_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64,
    :img_size => (28, 28),
    :img_channels => 1,
)
args[:imzprod] = prod(args[:img_size])

## =====

device!(0)

dev = gpu

##=====

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]
train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))


train_fashion, fashion_train_labels = FashionMNIST(; split=:train)[:]
test_fashion, fashion_test_labels = FashionMNIST(; split=:test)[:]
fashion_train_labels = Float32.(Flux.onehotbatch(fashion_train_labels, 0:9))
fashion_test_labels = Float32.(Flux.onehotbatch(fashion_test_labels, 0:9))

train_sdigits, train_slabels = SVHN2(; split=:train)[:]
train_sdigits = imresize(dropdims(Float32.(mean(Gray.(train_sdigits), dims=3)), dims=3), (28, 28))
train_slabels = Float32.(Flux.onehotbatch(train_slabels .- 1, 0:9))

test_sdigits, test_slabels = SVHN2(; split=:test)[:]
test_sdigits = imresize(dropdims(Float32.(mean(Gray.(test_sdigits), dims=3)), dims=3), (28, 28))
test_slabels = Float32.(Flux.onehotbatch(test_slabels .- 1, 0:9))

train_loader = DataLoader((dev(train_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)
test_loader = DataLoader((dev(test_digits)); batchsize=args[:bsz], shuffle=true,
    partial=false)

## =====

primary = Chain(
    flatten,
    SDense(784, 64, relu), Dense(64, 10)
)

x = train_sdigits[:, :, 1:64]

m = SDense(784, 64, relu)

f(x) = relu6(x) / 6.0f0

xx = -10:0.1:10
flatten(x)
m.W
m.W * flatten(x)

w = relu.(randn(Float32, 64, 784))
m.W

heatmap(w .* m.W)



H = Chain(
    Dense(10, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
)