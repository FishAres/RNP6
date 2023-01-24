using Flux, Zygote
using TensorBoardLogger, Logging
using JLD2
using BSON

## =====
"For multiple output heads"
struct Split{T}
    layers::T
end

Split(layers...) = Split(layers)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.layers)

## ======

dense_plus(in_sz, lsz; f=elu, l2=BatchNorm) = Chain(Dense(in_sz, lsz), l2(lsz, f))

function BasicBlock(channels::Pair{Int64,Int64}, connection; stride::Int64=1)
    layer = Chain(Conv((3, 3), channels; stride, pad=1, bias=false),
                  BatchNorm(channels[2], relu),
                  Conv((3, 3), channels[2] => channels[2]; pad=1, bias=false),
                  BatchNorm(channels[2]))
    return Chain(SkipConnection(layer, connection), x -> relu.(x))
end

## ==== saving

function save_model(model, savestring; local_=true)
    model = cpu(model)
    if local_
        full_str = "saved_models/" * savestring * ".bson"
    else
        full_str = savestring * ".bson"
    end
    BSON.@save full_str model
    return println("saved at $(full_str)")
end
