using LinearAlgebra, Statistics
using Plots
using StatsBase: sample
using ArgParse

function meshgrid(x, y)
    xs = x' .* ones(length(y))
    ys = ones(length(x))' .* y
    return xs, ys
end

rgb_to_gray(x) = 0.3 * x[:, :, 1] + 0.59 * x[:, :, 2] + 0.1 * x[:, :, 3]

function maprange(x, r1, r2)
    span1 = r1[end] - r1[1]
    span2 = r2[end] - r2[1]
    vs = @. (x - r1[1]) / span1
    @. r2[1] + (vs * span2)
end

rect(w, h, x, y) = Shape(x .+ [0, w, w, 0, 0], y .+ [0, 0, h, h, 0])

function partial(f, a...)
    return (b...) -> f(a..., b...)
end


"""
Short-hand for Zygote.ignore() block
"""
stopgrad(x) =
    Zygote.ignore() do
        return x
    end

"Parse command line arguments"
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_filters"
        help = "number of learned filters used"
        arg_type = Int
        default = 64
        "--device_id"
        help = "GPU device"
        default = 0
    end
    return parse_args(s)
end

function update_args_dict(model_args_dict, args)
    for key in keys(model_args_dict)
        args[Symbol(key)] = model_args_dict[key]
    end
    args
end