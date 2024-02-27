using DrWatson
@quickactivate "RNP6"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux
using Flux: batch, unsqueeze
using IterTools: partition
using Random: shuffle
using Plots

include(srcdir("gen_vae_utils.jl"))
## =====

args = Dict(
    :bsz => 64,
    :img_size => (28, 28),
    :img_channels => 1,
)

dev = gpu
## ======

train_digits, train_labels = MNIST(; split=:train)[:]
test_digits, test_labels = MNIST(; split=:test)[:]

train_vec = collect(eachslice(train_digits, dims=3))
test_vec = collect(eachslice(test_digits, dims=3))
train_batches = unsqueeze.(batch.(shuffle(collect(partition(train_vec, args[:bsz])))), 3)
test_batches = unsqueeze.(batch.(shuffle(collect(partition(test_vec, args[:bsz])))), 3)

## ======

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev

## ======

b = mnist_quadrants(train_digits)
heatmap(b[:, :, 1])



function mnist_quadrants(xs; args=args)
    tmp = collect(partition(eachslice(xs, dims=3), 4))
    a = map(x -> vcat(x[1:2]...), tmp)
    b = map(x -> vcat(x[3:4]...), tmp)
    c = map(x -> hcat(x...), zip(a, b))
    resized_vec = map(x -> imresize(x, args[:img_size]), c)
    return fast_img_concat(resized_vec)
end

function rand_mnist_quadrants(xs; args=args)
    n_digits = 2
    quad_ind = sample(1:4, 2, replace=false)

end

n_digits = 2
digit_pairs = collect(partition(collect(eachslice(train_digits, dims=3)), 2))

quad_ind = sample(1:4, 2, replace=false)
quad_zeros = setdiff(collect(1:4), quad_ind)
zeros_mat = zeros(Float32, 28, 28, 1)

function gen_quadrants(digit_pairs, ind)
    canv = [zeros_mat[:, :, 1] for _ in 1:4]
    canv[quad_ind] .= digit_pairs[ind]
    canv_2 = collect(partition(canv, 2))
    canv2 = map(x -> vcat(x...), canv_2)
    imresize(hcat(canv2...), args[:img_size])
end





function fast_img_concat(xs; args=args)
    cat_ims = zeros(Float32, args[:img_size]..., length(xs))
    Threads.@threads for i in 1:length(xs)
        cat_ims[:, :, i] = xs[i]
    end
    return cat_ims
end

train_data

heatmap(train_data[:, :, 13])

