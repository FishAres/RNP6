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

"""
Basic ResNet BasicBlock
#Arguments
- `channels::Pair{Int, Int}`: input and output channel dims
- `connection::function`: Function to apply to input and output (usually +)
"""
function BasicBlock(channels::Pair{Int64,Int64}, connection; stride::Int64=1)
    layer = Chain(Conv((3, 3), channels; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2]; pad=1, bias=false),
        BatchNorm(channels[2]))
    return Chain(SkipConnection(layer, connection), x -> relu.(x))
end

## ==== saving

# """
# To be deprecated. BSON.jl isn't great for saving models
# """
# function save_model(model, savestring; local_=true)
#     model = cpu(model)
#     full_str = savestring * ".bson"
#     BSON.@save full_str model
#     return println("saved at $(full_str)")
# end

function save_model(model, savestring)
    model_state = Flux.state(model |> cpu)
    jldsave(savestring * ".jld2"; model_state)
    println("saved at $savestring")
end

function load_model(model, savestring)
    ms = JLD2.load(savestring * ".jld2", "model_state")
    Flux.loadmodel!(model, ms)
end