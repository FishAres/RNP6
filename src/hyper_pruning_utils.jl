using Flux

function create_bias(weights::AbstractArray, bias::Bool, dims::Tuple...)
    return bias ? fill!(similar(weights, dims...), 0) : false
end

struct SDense
    W
    b
    σ
    SDense(weight, b, σ) = new(weight, b, σ)
end

function SDense(in::Integer, out::Integer, σ=identity;
    init=Flux.glorot_uniform, bias=true)
    w = init(out, in)
    b = create_bias(w, bias, (size(w, 1), size(w, 3)))
    return SDense(w, b, σ)
end

function (m::SDense)(x::AbstractArray, w::AbstractArray)
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    x = length(size(x)) > 2 ? x : unsqueeze(x, 2)
    b_ = isa(m.bias, AbstractArray) ? unsqueeze(m.bias, 2) : m.bias
    return σ.((w .* m.W) * x .+ b_)
end
