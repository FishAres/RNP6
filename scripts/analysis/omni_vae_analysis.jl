
function get_μ_zs(x)
    z1s = []
    μ, logvar = Encoder(x)
    θsz = H_state(μ)
    θsa = H_policy(μ)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push!(z1s, cpu(z1))
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push!(z1s, cpu(z1))
    end
    return μ |> cpu, z1s
end

## === TSne
using TSne, Clustering

begin
    z2s, z1s = [], []
    for (i, x) in enumerate(test_loader)
        μ, z1 = get_μ_zs(x)
        push!(z2s, μ)
        push!(z1s, hcat(z1...))
    end
end

using StatsBase: shuffle
function shuffle_batchwise(x1, x2)
    b = shuffle([x1; x2])
    return hcat(b...)
end

function sample_levels(z, x; args=args)
    θsz = H_state(z)
    θsa = H_policy(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    out_x̂ = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        out_x̂ += sample_patch(x̂, a1, sampling_grid)
    end
    return out |> cpu, out_x̂ |> cpu
end


Z = shuffle_batchwise(z2s, z1s)
sample_inds = sample(1:size(Z, 2), 4000, replace=false)

Ys = tsne(Z[:, sample_inds]')
scatter(Ys[:, 1], Ys[:, 2])

n_clusters = 10
R = kmeans(Ys', n_clusters)
inds = assignments(R)

begin
    p = plot(legend=false)
    for i in 1:n_clusters
        scatter!(Ys[inds.==i, 1], Ys[inds.==i, 2], c=i)
    end
    p
end
using Flux: batch
a = batch.(collect(partition(eachcol(Z[:, sample_inds]), args[:bsz]))) |> gpu
outputs, outputs_x̂ = let
    out_, out_x̂_ = [], []
    for z in a
        out, out_x̂ = sample_levels(z, x) |> cpu
        push!(out_, dropdims(out, dims=3))
        push!(out_x̂_, dropdims(out_x̂, dims=3))
    end
    out_, out_x̂_
end

outputs[1]
out_cat = cat(outputs..., dims=3)
out_cat_x̂ = cat(outputs_x̂..., dims=3)
trunc_inds = inds[1:size(out_cat, 3)]

ind = 0
b
begin
    ind += 1
    b = out_cat[:, :, trunc_inds.==ind]
    # b = out_cat_x̂[:, :, trunc_inds.==ind]
    stack_ims(b, n=8) |> plot_digit
end