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

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        return [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[3], relu6), flatten)

    Enc_e_z = Chain(x -> vcat(x...), HyDense(784 + args[:esz], args[:esz], Θ[4], elu),
        flatten)

    err_rnn = ps_to_RN(get_rn_θs(Θ[5], args[:esz], args[:esz]); f_out=fz)
    z0 = fz.(Θ[6])

    return (Enc_za_z, f_state, Dec_z_x̂, Enc_e_z, err_rnn), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        return [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)
    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function full_sequence(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a, Enc_e_z, err_rnn = models
    z1, a1, x̂, patch_t, err, h_err = forward_pass(z0, a0, models, x;
        scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t, err, h_err = forward_pass(z1, h_err, models, x;
            scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end

function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset], err_grad=false)
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a, Enc_e_z, err_rnn = models

    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    err = if err_grad
        flatten(patch_t) .- flatten(x̂)
    else
        stopgrad(flatten(patch_t) .- flatten(x̂))
    end
    h_err = err_rnn(Enc_e_z((err, err_minus1)))
    return z1, a1, x̂, patch_t, err, h_err
end

function forward_pass2(z1, a1, models, x; scale_offset=args[:scale_offset], err_grad=false)
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a, Enc_e_z, err_rnn = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))
    return z1, a1, patch_t
end

function get_models(θsz, θsa; args=args, Hx_bounds=Hx_bounds, Ha_bounds=Ha_bounds)
    (Enc_za_z, f_state, Dec_z_x̂, Enc_e_z, err_rnn), z0 = get_fstate_models(θsz, Hx_bounds;
        args=args)
    (Enc_za_a, f_policy, Dec_z_a), a0 = get_fpolicy_models(θsa, Ha_bounds; args=args)
    models = f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a, Enc_e_z, err_rnn
    return models, z0, a0
end

function get_models(z; args=args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return models, z0, a0
end

function model_forward(z, x; args=args)
    models, z0, a0 = get_models(z)
    z1, a1, x̂, patch_t, err, h_err = forward_pass(z0, a0, models, x;
        scale_offset=args[:scale_offset])
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t, err, h_err = forward_pass(z1, a1, models, x;
            scale_offset=args[:scale_offset])
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, h_err
end

function model_forward2(z_init, x; err_grad=true)
    models, z0, a0 = get_models(z_init)
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a, Enc_e_z, err_rnn = models
    z1, a1, patch_t = forward_pass2(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small, h_err = model_forward(z1, patch_t)
    err = if err_grad
        flatten(patch_t) .- flatten(out_small)
    else
        stopgrad(flatten(patch_t) .- flatten(out_small))
    end
    out = sample_patch(out_small, a1, sampling_grid)
    h_err2 = err_rnn(Enc_e_z((err, h_err)))
    for t in 2:args[:seqlen]
        z1, a1, patch_t = forward_pass2(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small, h_err = model_forward(z1, patch_t)
        err = if err_grad
            flatten(patch_t) .- flatten(out_small)
        else
            stopgrad(flatten(patch_t) .- flatten(out_small))
        end
        h_err2 = err_rnn(Enc_e_z((err, h_err)))
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return out, h_err2
end

function model_loss(z_init, x, rs)
    Flux.reset!(RN2)
    out_full, h_err2 = model_forward2(z_init, x)
    μ, logvar = RN2(h_err2)
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out_full), flatten(x); agg=sum)
    z = sample_z(μ, logvar, rs[1])
    for t in 2:args[:seqlen]
        out_full, h_err2 = model_forward2(z, x)
        μ, logvar = RN2(h_err2)
        z = sample_z(μ, logvar, rs[t])
        klqp += kl_loss(μ, logvar)
        rec_loss += Flux.mse(flatten(out_full), flatten(x); agg=sum)
    end
    return rec_loss, klqp
end

function get_loop(x)
    full_recs, zs = [], []
    rs = gpu([randn(Float32, args[:π], args[:bsz]) for _ in 1:args[:seqlen]])
    z_init = gpu(randn(Float32, args[:π], args[:bsz]))
    Flux.reset!(RN2)
    out_full, h_err2 = model_forward2(z_init, x)
    μ, logvar = RN2(h_err2)
    z = sample_z(μ, logvar, rs[1])
    push!(full_recs, cpu(out_full))
    push!(zs, cpu(z))
    for t in 2:args[:seqlen]
        out_full, h_err2 = model_forward2(z, x)
        push!(full_recs, cpu(out_full))
        push!(zs, cpu(z))
        μ, logvar = RN2(h_err2)
        z = sample_z(μ, logvar, rs[t])
    end
    return full_recs, zs
end 

function plot_rec(x, full_recs, ind)
    p = plot([plot_digit(full_rec[:, :, 1, ind]) for full_rec in full_recs]...)
    px = plot_digit(cpu(x)[:, :, ind])
    return plot(px, p)
end

function plot_recs(x, inds; args=args)
    full_recs, zs = get_loop(x)
    p = [plot_rec(x, full_recs, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = gpu([[rand(D, args[:π], args[:bsz]) for _ in 1:args[:seqlen]]
              for
              _ in 1:length(train_data)])
    for (i, x) in enumerate(train_data)
        z_init = gpu(randn(Float32, args[:π], args[:bsz]))
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(z_init, x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                return log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    rs = gpu([[rand(D, args[:π], args[:bsz]) for _ in 1:args[:seqlen]]
              for
              _ in 1:length(test_data)])
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        z_init = gpu(randn(Float32, args[:π], args[:bsz]))
        rec_loss, klqp = model_loss(z_init, x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

## ==== misc

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, x) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    return x_
end
