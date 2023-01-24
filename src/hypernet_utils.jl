using Flux

## ==== some functions

"get parameter sizes"
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

function create_bias(weights::AbstractArray, bias::Bool, dims::Tuple...)
    return bias ? fill!(similar(weights, dims...), 0) : false
end
function split_bias(in::Integer, out::Integer, W::AbstractMatrix)
    ab, bsz = size(W)
    in_out = (in * out)
    bsize = ab - in_out
    W_ = reshape(W[1:in_out, :], out, in, :)
    b_ = bsize > 0 ? W[(in_out+1):end, :] : false
    return W_, b_
end

##==== Dense

struct HyDense
    weight::Any
    bias::Any
    σ::Any
    HyDense(weight, b, σ) = new(weight, b, σ)
end
Flux.@functor HyDense

function HyDense(in::Integer, out::Integer, bsz::Integer, σ=identity;
    init=Flux.glorot_uniform, bias=true)
    w = init(out, in, bsz)
    b = create_bias(w, bias, (size(w, 1), size(w, 3)))
    return HyDense(init(out, in, bsz), b, σ)
end

"single parameter matrix W, known input-output dims"
function HyDense(in::Integer, out::Integer, W::AbstractMatrix, σ=identity)
    W_, b_ = split_bias(in, out, W)
    return HyDense(W_, b_, σ)
end

function (m::HyDense)(x::AbstractArray)
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    x = length(size(x)) > 2 ? x : unsqueeze(x, 2)
    b_ = isa(m.bias, AbstractArray) ? unsqueeze(m.bias, 2) : m.bias
    return σ.(batched_mul(m.weight, x) .+ b_)
end

function Base.show(io::IO, l::HyDense)
    print(io,
        "HyDense(",
        size(l.weight, 2),
        " => ",
        size(l.weight, 1),
        ", batch size: ",
        size(l.weight, 3))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    return print(io, ")")
end

## === Conv

struct HyConv
    weight::Any
    bias::Any
    σ::Any
    stride::Any
    pad::Any
    groups::Any
    HyConv(weight, b, σ, stride, pad, groups) = new(weight, b, σ, stride, pad, groups)
end
Flux.@functor HyConv

function split_bias_conv(kernelsize::Tuple, in_channels::Integer, out_channels::Integer,
    W::AbstractMatrix)
    bsize = out_channels
    W_ = reshape(W[1:(end-bsize), :], kernelsize..., in_channels, out_channels, :)
    b_ = bsize > 0 ? W[(end-bsize+1):end, :] : false
    return W_, b_
end

function batched_conv(w, x, stride, pad)
    return cat([conv(a, b; stride=stride, pad=pad)
                for
                (a, b) in zip(eachslice(x; dims=5), eachslice(w; dims=5))]...;
        dims=4)
end

"initialize without weights"
function HyConv(kernelsize::Tuple,
    in_channels::Integer,
    out_channels::Integer,
    batchsize::Integer,
    σ=identity,
    init=Flux.glorot_uniform,
    bias=true,
    stride=(1, 1),
    pad=(0, 0))
    w = init(kernelsize..., in_channels, out_channels, batchsize)
    b = create_bias(w, true, (size(w, 4), size(w, 5)))
    return HyConv(w, b, σ, stride, pad)
end

"single parameter matrix W, known input-output dims"
function HyConv(kernelsize::Tuple,
    in_channels::Integer,
    out_channels::Integer,
    W::AbstractMatrix,
    σ=identity,
    stride=(1, 1),
    pad=(0, 0))
    W_, b_ = split_bias_conv(kernelsize, in_channels, out_channels, W)
    return HyConv(W_, b_, σ, stride, pad)
end

function (m::HyConv)(x::AbstractArray)
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    x = length(size(x)) > 4 ? x : unsqueeze(x, 4)
    b_ = isa(m.bias, AbstractArray) ? unsqueeze(unsqueeze(m.bias, 1), 1) : m.bias
    return σ.(batched_conv(m.weight, x, m.stride, m.pad) .+ b_)
end

## ==== conv transpose
using NNlib: ∇conv_data
import Flux: conv_transpose_dims, calc_padding, expand

struct HyConvTranspose
    weight::Any
    bias::Any
    σ::Any
    stride::Any
    pad::Any
    dilation::Any
    groups::Any
    # HyConvTranspose(weight, b, σ, stride, pad, dilation, groups) = new(weight, b, σ, stride, pad, dilation, groups)
end
Flux.@functor HyConvTranspose

function conv_transpose_dims(c::HyConvTranspose, x::AbstractArray)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
    I = (size(x)[1:(end-2)] .- 1) .* c.stride .+ 1 .+
        (size(c.weight)[1:(end-3)] .- 1) .* c.dilation .- combined_pad
    C_in = size(c.weight)[end-2] * c.groups
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    w_size = size(c.weight)[1:(end-1)]
    return DenseConvDims((I..., C_in, batch_size),
        w_size;
        stride=c.stride,
        padding=c.pad,
        dilation=c.dilation,
        groups=c.groups)
end

function split_bias_conv_transpose(kernelsize::Tuple, in_channels::Integer,
    out_channels::Integer, W::AbstractMatrix)
    bsize = out_channels
    W_ = reshape(W[1:(end-bsize), :], kernelsize..., out_channels, in_channels, :)
    b_ = bsize > 0 ? W[(end-bsize+1):end, :] : false
    return W_, b_
end

function batched_conv_transpose(x, w, cdims)
    return cat([∇conv_data(a, b, cdims)
                for
                (a, b) in zip(eachslice(x; dims=5), eachslice(w; dims=5))]...;
        dims=4)
end

"initialize without weights"
function HyConvTranspose(kernelsize::Tuple,
    channels::Pair,
    batchsize::Integer,
    σ=identity;
    init=Flux.glorot_uniform,
    bias=true,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
    groups=1)
    w = init(kernelsize..., channels[2], channels[1], batchsize)
    b = create_bias(w, bias, (size(w, 3), size(w, 5)))
    # HyConvTranspose(w, b, σ, stride, pad, dilation, groups)
    return HyConvTranspose(w, b, σ; stride, pad, dilation, groups)
end

"single parameter matrix W, known input-output dims"
function HyConvTranspose(W::AbstractArray{T,N}, b, σ=identity; stride=1, pad=0, dilation=1,
    groups=1) where {T,N}
    stride = expand(Val(N - 3), stride)
    # println(N, " ", stride)
    dilation = expand(Val(N - 3), dilation)
    pad = calc_padding(ConvTranspose, pad, size(W)[1:(N-3)], dilation, stride)
    # W_, b_ = split_bias_conv(kernelsize, in_channels, out_channels, W)
    return HyConvTranspose(W, b, σ, stride, pad, dilation, groups)
end

function HyConvTranspose(kernelsize::Tuple,
    channels::Pair,
    W::AbstractMatrix,
    σ=identity;
    stride=1,
    pad=0,
    dilation=1,
    groups=1)
    stride = expand(Val(5 - 3), stride)
    # println(N, " ", stride)
    dilation = expand(Val(5 - 3), dilation)
    pad = calc_padding(ConvTranspose, pad, size(W)[1:(5-3)], dilation, stride)
    W_, b_ = split_bias_conv_transpose(kernelsize, channels[1], channels[2], W)
    return HyConvTranspose(W_, b_, σ, stride, pad, dilation, groups)
end

function (m::HyConvTranspose)(x::AbstractArray)
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    b_ = isa(m.bias, AbstractArray) ? unsqueeze(unsqueeze(m.bias, 1), 1) : m.bias
    cdims = conv_transpose_dims(m, x)
    x = length(size(x)) > 4 ? x : unsqueeze(x, 4)
    return σ.(batched_conv_transpose(x, m.weight, cdims) .+ b_)
end

## == RNNs
@inline bmul(a, b) = dropdims(batched_mul(unsqueeze(a, 1), b); dims=1)

"RNN parameterized by Wh, Wx, b, with initial state h"
function RN(Wh, Wx, b, h, x; f=elu)
    h = f.(bmul(h, Wh) + bmul(x, Wx) + b)
    return h, h
end

gate(h, n) = (1:h) .+ h * (n - 1)
gate(x::AbstractVector, h, n) = @view x[gate(h, n)]
gate(x::AbstractMatrix, h, n) = @view x[gate(h, n), :]

function gru_output(Wx, Wh, b, x, h)
    o = size(h, 1)
    gx = bmul(x, Wx)
    gh = bmul(h, Wh)
    r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    return gx, gh, r, z
end

function manzGRU(Wh, Wx, b, h, x)
    b, o = b, size(h, 1)
    gx, gh, r, z = gru_output(Wx, Wh, b, x, h)
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(x)
    return h′, reshape(h′, :, sz[2:end]...)
end

"(θ: b, Wh, Wx, h) -> Recur(RNN(..., h))"
function ps_to_RN(θ; rn_fun=RN, args=args, f_out=elu)
    Wh, Wx, b, h = θ
    ft = (h, x) -> rn_fun(Wh, Wx, b, h, x; f=f_out)
    return Flux.Recur(ft, h)
end

"size for RNN Wh, Wx, b"
get_rnn_θ_sizes(esz, hsz) = esz * hsz + hsz^2 + 2 * hsz

function rec_rnn_θ_sizes(esz, hsz)
    return [hsz^2, esz * hsz, hsz, hsz]
end

function get_rn_θs(rnn_θs, esz, hsz)
    fx_sizes = rec_rnn_θ_sizes(esz, hsz)
    @assert sum(fx_sizes) == size(rnn_θs, 1)
    fx_inds = [0; cumsum(fx_sizes)]
    # split rnn_θs vector according to cumulative fx_inds
    Wh_, Wx_, b, h = collect(rnn_θs[(fx_inds[ind-1]+1):fx_inds[ind], :]
                             for ind in 2:length(fx_inds))
    Wh = reshape(Wh_, hsz, hsz, size(Wh_)[end])
    Wx = reshape(Wx_, esz, hsz, size(Wx_)[end])
    return Wh, Wx, b, h
end

## maybe useful soon

function get_offsets(m::Flux.Chain)
    θ, re = Flux.destructure(m)
    m_offsets = re.offsets
    offsets_ = []
    function get_offsets_(offsets, offsets_)
        for mo in offsets
            for key in keys(mo)
                a = mo[key]
                if isa(a, Number)
                    push!(offsets_, a)
                elseif isa(a, NamedTuple) || isa(a, Tuple) && !isempty(a)
                    get_offsets_(a, offsets_)
                end
            end
        end
        return offsets_
    end
    return get_offsets_(m_offsets, offsets_)
end
