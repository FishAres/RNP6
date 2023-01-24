using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Flux: unsqueeze
using Images

const π32 = Float32(π)

function filter_batch(x, f, p::Number=1.0f0)
    sz = size(x)
    x = length(sz) > 3 ? dropdims(x; dims=3) : x
    x = collect(eachslice(cpu(x); dims=3))
    out = cat([imfilter(k[:, :, 1], f(p)) for k in x]...; dims=3)
    return out = length(sz) > 3 ? unsqueeze(out, 3) : out
end

"generate sampling grid 3 x (width x height) x (batch size)"
function get_sampling_grid(width, height; args=args)
    x = LinRange(-1, 1, width)
    y = LinRange(-1, 1, height)
    x_t_flat = reshape(repeat(x, height), 1, height * width)
    y_t_flat = reshape(repeat(transpose(y), width), 1, height * width)
    all_ones = ones(eltype(x_t_flat), 1, size(x_t_flat)[2])
    sampling_grid = vcat(x_t_flat, y_t_flat, all_ones)
    sampling_grid = reshape(transpose(repeat(transpose(sampling_grid), args[:bsz])),
        3,
        size(x_t_flat, 2),
        args[:bsz])
    return Float32.(sampling_grid)
end

function get_rot_mat(theta_rot)
    cos_rot = reshape(cos.(theta_rot), 1, 1, :)
    sin_rot = reshape(sin.(theta_rot), 1, 1, :)
    return hcat(vcat(cos_rot, -sin_rot), vcat(sin_rot, cos_rot))
end

function get_shear_mat(theta_shear)
    theta_shear = reshape(theta_shear, 1, 1, :)
    return hcat(vcat(ones_vec, theta_shear), vcat(zeros_vec, ones_vec))
end

@inline function get_affine_mats(thetas; scale_offset=0.0f0)
    sc = (@view thetas[[1, 4], :]) .+ 1.0f0 .+ scale_offset
    b = sc .* (@view thetas[5:6, :])
    A_rot = get_rot_mat(π32 * (@view thetas[2, :]))
    A_sc = unsqueeze(sc, 2) .* diag_mat
    A_shear = get_shear_mat(@view thetas[3, :])
    return A_rot, A_sc, A_shear, b
end

function grid_generator_3d(sampling_grid_2d, thetas; scale_offset=args[:scale_offset])
    A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=scale_offset)
    A = batched_mul(batched_mul(A_rot, A_shear), A_s)
    return batched_mul(A, sampling_grid_2d) .+ unsqueeze(b, 2)
end

"add offset to the diagonal of a square matrix
to make it safe for inversion
"
Zygote.@nograd function safe_inv(A; offset=1e-5)
    T = typeof(A[1])
    offs_ = T(offset)
    thresh_ = T(1e-5)
    As = copy(A)
    for i in diagind(A)
        if As[i] < thresh_
            As[i] += offs_
        end
    end
    return inv(As)
end

"faster to do one matrix inverse but needs (const) diag_off(set) for stability"
Zygote.@nograd function get_inv_grid(sampling_grid_2d, thetas;
    scale_offset=args[:scale_offset])
    A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=scale_offset)
    A = batched_mul(batched_mul(A_rot, A_shear), A_s) #.+ diag_off
    # A = batched_mul(batched_mul(A_rot, A_shear), A_s)
    # sh_inv = cat(map(inv, eachslice(cpu(A_shear), dims=3))..., dims=3)
    # sc_inv = cat(map(inv, eachslice(cpu(A_s) .+ 1.0f-5, dims=3))..., dims=3)
    # rot_inv = cat(map(inv, eachslice(cpu(A_rot), dims=3))..., dims=3)
    # Ainv = batched_mul(batched_mul(sc_inv, sh_inv), rot_inv) |> gpu
    # Ainv = mul(sc_inv, sh_inv, rot_inv) |> gpu
    Ainv = gpu(cat(map(safe_inv, eachslice(cpu(A); dims=3))...; dims=3))
    return batched_mul(Ainv, (sampling_grid_2d .- unsqueeze(b, 2)))
end

function sample_patch(x,
    thetas,
    sampling_grid;
    args=args,
    sz=args[:img_size],
    scale_offset=args[:scale_offset])
    grid = grid_generator_3d(sampling_grid, thetas; scale_offset=scale_offset)
    # grid = grid_generator_fast(sampling_grid, thetas)
    tg = reshape(grid, 2, sz..., size(grid)[end])
    x = reshape(x, sz..., args[:img_channels], size(x)[end])
    return grid_sample(x, tg; padding_mode=:zeros)
end

function sample_patch(x, transformed_grid; sz=args[:img_size])
    tg = reshape(transformed_grid, 2, sz..., size(transformed_grid)[end])
    x = reshape(x, sz..., args[:img_channels], size(x)[end])
    return grid_sample(x, tg; padding_mode=:zeros)
end

Zygote.@nograd function zoom_in2d(x, xy, sampling_grid; args=args,
    scale_offset=args[:scale_offset])
    inv_grid = get_inv_grid(sampling_grid, xy; scale_offset=scale_offset)
    out_inv = sample_patch(x, inv_grid)
    return out_inv
end
