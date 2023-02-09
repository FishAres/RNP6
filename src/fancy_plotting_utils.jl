using DrWatson
using Images

include(srcdir("utils.jl"))

function orange_on_rgb(x, a, b, c)
    # a_ = a .+ x
    # b_ = b .+ 0.65 .* x
    # c_ = c
    a_ = a
    b_ = b .+ 0.65 .* x
    c_ = c .+ x
    return a_, b_, c_
end

function orange_on_rgb_array(xs)
    xs_transpose = [batch([x' for x in eachslice(dropdims(z; dims=3); dims=3)]) for z in xs]
    xs_transpose = unsqueeze.(xs_transpose, 3)
    tmp_sum = orange_on_rgb(xs_transpose[end], xs_transpose[1:3]...)
    tmp_sum = cat(tmp_sum...; dims=3)
    ims = [permutedims(x, [3, 1, 2]) for x in eachslice(tmp_sum; dims=4)]
    return ims
end

# function stack_ims(xs; n=8)
#     n = n === nothing ? sqrt(length(xs)) : n
#     rows_ = map(x -> hcat(x...), partition(xs, n))
#     return cat(rows_...; dims=3)
# end

function stack_ims(xs; n=8)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end

function impermute(x)
    a, b, c = size(x)
    if a < b && a < c
        return permutedims(x, [2, 3, 1])
    else
        return permutedims(x, [3, 1, 2])
    end
end

@inline function sample_patch_rgb(x, xy, sampling_grid; sc_offset=args[:scale_offset],
    sz=args[:img_size])
    ximg = reshape(x, sz..., 3, size(x)[end])
    sc_ = maprange(view(xy, 1:2, :), -1.0f0:1.0f0, sc_offset)
    xy_ = view(xy, 3:4, :)
    tr_grid = affine_grid_generator(sampling_grid, sc_, xy_)
    return grid_sample(ximg, tr_grid; padding_mode=:zeros)
end

function full_sequence_patches(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    patches, xys = [], []
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    push!(patches, cpu(x̂))
    push!(xys, cpu(a1))
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
        push!(patches, cpu(x̂))
        push!(xys, cpu(a1))
    end
    return patches, out, xys
end

function full_sequence_patches(z::AbstractArray, x; args=args,
    scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence_patches(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

function get_loop_patches(x; args=args)
    outputs = patches, recs, as, as1 = [], [], [], [], []
    r = gpu(rand(args[:D], args[:π], args[:bsz]))
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out1_patches, out_small, xys_1 = gpu(full_sequence_patches(z1, patch_t))
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out_small, out1_patches, a1, xys_1), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out1_patches, out_small, xys_1 = gpu(full_sequence_patches(z1, patch_t))
        # out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out_small, out1_patches, a1, xys_1), outputs)
    end
    return outputs
end

function sample_patches(model, z; args=args, patch=args[:patch])
    patches, preds, xys, xys_1 = [], [], [], []
    function push_to_patches!(x̂, xy, out)
        push!(patches, cpu(x̂))
        push!(preds, cpu(out))
        return push!(xys, cpu(xy))
    end

    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)
    patch_t = Zygote.ignore() do
        return flatten(zoom(x, xyt, "in", sampling_grid))
    end
    out1_patches, out_1, xys1 = model_forward_patches(model, ẑ; patch=patch)
    out = sample_patch(out_1, xyt, sampling_grid; sc_offset=args[:scale_offset])
    push_to_patches!(out_1, xyt, out1_patches)
    push!(xys_1, xys1)
    @inbounds for t in 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        x̂ = Dx(ẑ)
        patch_t = Zygote.ignore() do
            return flatten(zoom(x, xyt, "in", sampling_grid))
        end
        out1_patches, out_1, xys1 = model_forward_patches(model, ẑ; patch=patch)
        out += sample_patch(out_1, xyt, sampling_grid; sc_offset=args[:scale_offset])
        push_to_patches!(out_1, xyt, out1_patches)
        push!(xys_1, xys1)
    end
    return patches, preds, xys, xys_1
end

function map_to_rgb(a, b, c, d)
    rgb_arrays = [zeros(3, 28, 28) for _ in 1:4]
    rgb_arrays[3][1, :, :] .= reshape(a .+ 0.2f0, 28, 28)'
    rgb_arrays[2][2, :, :] .= reshape(b, 28, 28)'
    rgb_arrays[1][3, :, :] .= reshape(c, 28, 28)'

    # 4th element is orange
    rgb_arrays[4][1, :, :] .= reshape(d, 28, 28)'
    rgb_arrays[4][2, :, :] .= 0.6 .* reshape(d, 28, 28)'

    return rgb_arrays
end

function get_digit_parts(xs, batchind; savestring=nothing, format_=".png", resize_=(80, 80))
    patches, sub_patches, xys, xys1 = gpu(get_loop_patches(xs))
    pasted_patches = cpu([sample_patch(patches[i], xys[i], sampling_grid)
                          for i in 1:length(patches)])

    digits = orange_on_rgb_array(pasted_patches)
    # digit_patches = [orange_on_rgb_array(map(x -> reshape(x, 28, 28, 1, 64), z)) for z in sub_patches]

    # digit_patches = [
    # orange_on_rgb_array([
    # sample_patch(sub_patch, xy, sampling_grid) for
    # (sub_patch, xy) in zip(subpatch_vec, xy_vec)
    # ]) for (subpatch_vec, xy_vec) in zip(sub_patches, xys1)
    # ] |> cpu

    digit_patches = cpu([orange_on_rgb_array([sample_patch(sample_patch(sub_patch, xy,
            sampling_grid), xy2,
        sampling_grid)
                                              for
                                              (sub_patch, xy, xy2) in
                                              zip(subpatch_vec, xy_vec, xys)])
                         for (subpatch_vec, xy_vec) in zip(sub_patches, xys1)])

    digit_im = map(clamp01nan, imresize(colorview(RGB, digits[batchind]), resize_))
    patch_ims = [map(clamp01nan, imresize(colorview(RGB, patch_[batchind]), resize_))
                 for
                 patch_ in digit_patches]

    if savestring !== nothing
        save(savestring * format_, digit_im)
        [save(savestring * "_part_$i" * format_, patch_)
         for
         (i, patch_) in enumerate(patch_ims)]
    end
    return digits, digit_patches, digit_im, patch_ims, xys, xys1
end
