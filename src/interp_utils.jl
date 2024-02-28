using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Flux: unsqueeze
using Images

const π32 = Float32(π) # π 😺

"""
Generate sampling grid (3 x (width x height) x (batch size)).
You only need to do this once
"""
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

"Generate rotation matrix for angle `theta_rot"
function get_rot_mat(theta_rot)
    cos_rot = reshape(cos.(theta_rot), 1, 1, :)
    sin_rot = reshape(sin.(theta_rot), 1, 1, :)
    return hcat(vcat(cos_rot, -sin_rot), vcat(sin_rot, cos_rot))
end

"Generate shear matrix for shear parameter `theta_shear`"
function get_shear_mat(theta_shear)
    theta_shear = reshape(theta_shear, 1, 1, :)
    return hcat(vcat(ones_vec, theta_shear), vcat(zeros_vec, ones_vec))
end

"""
Generate all affine transformation matrices
Note: `diag_mat` needs to be a `const` in scope
"""
function get_affine_mats(thetas; scale_offset=0.0f0)
    # scale
    sc = (@view thetas[[1, 4], :]) .+ 1.0f0 .+ scale_offset
    A_sc = unsqueeze(sc, 2) .* diag_mat
    # offset
    b = sc .* (@view thetas[5:6, :])
    A_rot = get_rot_mat(π32 * (@view thetas[2, :]))
    A_shear = get_shear_mat(@view thetas[3, :])
    return A_rot, A_sc, A_shear, b
end

"""
Affine transform the sampling grid
#Arguments
- `sampling_grid_2d::Array{Float, 3}`: Sampling grid with first two spatial dimensions only
- `thetas::Array`: Affine parameters
- `scale_offset::Float`: Scaling parameter
"""
function grid_generator(sampling_grid_2d, thetas; scale_offset=args[:scale_offset])
    A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=scale_offset)
    A = batched_mul(batched_mul(A_rot, A_shear), A_s)
    return batched_mul(A, sampling_grid_2d) .+ unsqueeze(b, 2)
end

"""
Add offset to the diagonal of a square matrix
to make it safe for inversion
"""
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

"""
Inverse affine transform of the sampling grid
It's faster to do one matrix inverse but has issues with stability
Note: this automatically uses the gpu
"""
Zygote.@nograd function get_inv_grid(sampling_grid_2d, thetas;
    scale_offset=args[:scale_offset])
    A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=scale_offset)
    A = batched_mul(batched_mul(A_rot, A_shear), A_s)
    Ainv = gpu(cat(map(safe_inv, eachslice(cpu(A); dims=3))...; dims=3))
    return batched_mul(Ainv, (sampling_grid_2d .- unsqueeze(b, 2)))
end

"""
Sample `x` at new grid transformed by parameters `thetas`
"""
function sample_patch(x,
    thetas,
    sampling_grid;
    args=args,
    sz=args[:img_size],
    scale_offset=args[:scale_offset])
    grid = grid_generator(sampling_grid, thetas; scale_offset=scale_offset)
    grid_r = reshape(grid, 2, sz..., :) # reshape grid
    x_r = reshape(x, sz..., args[:img_channels], :) # reshape x
    return grid_sample(x_r, grid_r; padding_mode=:zeros)
end

"""
Sample `x` at `transformed_grid`
"""
function sample_patch(x, transformed_grid; sz=args[:img_size])
    tg = reshape(transformed_grid, 2, sz..., size(transformed_grid)[end])
    x = reshape(x, sz..., args[:img_channels], size(x)[end])
    return grid_sample(x, tg; padding_mode=:zeros)
end

"""
Sample `x` with grid "inverse-transformed" by `thetas`
"""
Zygote.@nograd function zoom_in2d(x, thetas, sampling_grid; args=args,
    scale_offset=args[:scale_offset])
    inv_grid = get_inv_grid(sampling_grid, thetas; scale_offset=scale_offset)
    out_inv = sample_patch(x, inv_grid)
    return out_inv
end
