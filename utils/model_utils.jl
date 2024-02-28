
"""
Generate RNP outputs from the program vector z
"""
function sample_(z, x; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return out |> cpu
end

"""
Stack the matrices in "xs" into an (n x n) array
"""
function stack_ims(xs::Vector[Array]; n=nothing)
    n = n === nothing ? sqrt(length(xs)) : n
    xs = length(size(xs)) > 3 ? dropdims(xs, dims=3) : xs
    xs = collect(eachslice(xs, dims=3))
    rows_ = map(x -> hcat(x...), partition(xs, n))
    return vcat(rows_...)
end

# ==== Plotting RNP sequences ====

"""
Have "grayscale" array x render as orange in RGB
"""
function rgb_to_orange(x)
    a = x
    b = 0.65f0 .* x
    c = 0.0f0 .* x
    cat(a, b, c, dims=3)
end


function orange_on_rgb(xs::Vector[Array])
    orange_patch = rgb_to_orange(xs[end])
    min.(cat(xs[1:3]..., dims=3) .+ orange_patch, 1.0f0)
end

"""
From a vector of 4 patches generated by an RNP,
create an RGB image where the 4th patch is orange
"""
function view_patches_rgb(patches::Vector[Array], ind)
    @assert length(patches) == 4
    im_array = orange_on_rgb(patches)
    colorview(RGB, permutedims(im_array[:, :, :, ind], [3, 1, 2]))
end