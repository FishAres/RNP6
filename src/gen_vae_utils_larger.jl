# gen_vae_utils.jl but with bigger (> 1 layer) (z, a) -> RNN encoder
using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Distributions
using ProgressMeter
using ProgressMeter: Progress
using Plots

include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))

include(srcdir("utils.jl"))
## ======

sample_z(μ, logvar, r) = μ + r .* (exp.(logvar))

kl_loss(μ, logvar) = sum(@. (exp(logvar) + μ^2 - logvar - 1.0f0))

"As of RNP6, Dec_z_x̂ has 2 layers by default"
function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        return [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(
        HyDense(args[:π] + args[:asz], 64, Θ[1], elu),
        flatten,
        HyDense(64, args[:esz], Θ[2], elu),
        flatten
    )

    f_state = ps_to_RN(get_rn_θs(Θ[3], args[:esz], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(
        HyDense(args[:π], 64, Θ[4], elu),
        flatten,
        HyDense(64, 784, Θ[5], relu6),
        flatten)

    z0 = fz.(Θ[6])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        return [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_a = Chain(
        HyDense(args[:π] + args[:asz], 64, Θ[1], elu),
        HyDense(64, args[:esz], Θ[2], elu),
        flatten
    )
    f_policy = ps_to_RN(get_rn_θs(Θ[3], args[:esz], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[4], sin), flatten)
    a0 = sin.(Θ[5])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function get_models(θsz, θsa; args=args, Hx_bounds=Hx_bounds, Ha_bounds=Ha_bounds)
    (Enc_za_z, f_state, Dec_z_x̂), z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    (Enc_za_a, f_policy, Dec_z_a), a0 = get_fpolicy_models(θsa, Ha_bounds; args=args)
    models = f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a
    return models, z0, a0
end

function get_models(z; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return models, z0, a0
end

"one iteration"
function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    return z1, a1, x̂, patch_t
end

function full_sequence(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end

function full_sequence(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end

"output sequence: full recs, local recs, xys (a1), patches_t"
function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    r = gpu(rand(args[:D], args[:π], args[:bsz]))
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D], clip_grads=false)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)])
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                return log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        clip_grads && foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)])
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

function plot_rec(out, x, ind; kwargs...)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    return plot(p1, p2, kwargs...)
end

function plot_rec(out, x, xs, ind)
    out_ = reshape(cpu(out), 28, 28, :)
    x_ = reshape(cpu(x), 28, 28, :)
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    p3 = plot([plot_digit(x[:, :, 1, ind]; boundc=false) for x in xs]...)
    return plot(p1, p2, p3; layout=(1, 3))
end

function plot_recs(x, inds; plot_seq=true, args=args)
    full_recs, patches, xys, patches_t = get_loop(x)
    p = if plot_seq
        let
            patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
            [plot_rec(full_recs[end], x, patches_, ind) for ind in inds]
        end
    else
        [plot_rec(full_recs[end], x, ind) for ind in inds]
    end

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, x) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    return x_
end
