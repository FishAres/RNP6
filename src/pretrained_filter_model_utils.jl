
## ==== Functions for sorting learned weights by dot product
function get_weight_dotprods(weights::Array{Float32,3})
    ws = collect(eachslice(weights, dims=3))
    dots = zeros(length(ws), length(ws))
    for i in 1:length(ws)
        for j in 1:length(ws)
            dots[i, j] = dot(ws[i], ws[j])
        end
    end
    return dots
end

function sort_weight_dotprods(weights; k=5)
    dotprods = get_weight_dotprods(weights)
    dp = triu(dotprods, 1)
    dp[dp.==0] .= NaN
    dp_sorted = sortperm(abs.(dp[:]))
    linear_inds = LinearIndices(dp)

    min_dotprod_inds = [findall(x -> x == i, linear_inds)[1].I for i in dp_sorted[1:k]]
    return min_dotprod_inds
end

function mean_dotprod_ind(dp, i)
    mean([dp[:, i]; dp[i, :]])
end

function get_uncorr_filters(weights; n_filters=10, resize=false)
    dotprods = get_weight_dotprods(weights)
    dp = triu(dotprods, 1)
    mean_dotprods = [mean_dotprod_ind(dp, i) for i in 1:size(dp, 1)]
    sorted_inds = sortperm(mean_dotprods)[1:n_filters]

    ws = weights[:, :, sorted_inds]
    ws = resize ? imresize(ws, (28, 28)) : ws
    # return ws |> flatten
    return ws
end

## ==== model functions

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        return [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(
        HyDense(args[:π], args[:n_filters], Θ[3], elu),
        flatten,
    )

    z0 = fz.(Θ[4])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        return [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:esz], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:esz], args[:ha_sz]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:ha_sz], args[:asz], Θ[3], sin), flatten)
    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function get_models(z; args=args)
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa, Hx_bounds, Ha_bounds; args=args)
    return models, z0, a0
end

function get_models(θsz, θsa, Hx_bounds, Ha_bounds; args=args)
    (Enc_za_z, f_state, Dec_z_x̂), z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    (Enc_za_a, f_policy, Dec_z_a), a0 = get_fpolicy_models(θsa, Ha_bounds; args=args)
    models = f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a
    return models, z0, a0
end

function dec_z_x̂(z, Dec_z_x̂, dec_filters)
    h = Dec_z_x̂(z)
    relu.(dec_filters * h)
end

"one iteration"
function forward_pass(z1, a1, models, x, dec_filters; scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = dec_z_x̂(z1, Dec_z_x̂, dec_filters)
    patch_t = flatten(zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset))

    return z1, a1, x̂, patch_t
end

function full_sequence(models::Tuple, z0, a0, x, dec_filters; args=args,
    scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x, dec_filters; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x, dec_filters; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end

function full_sequence(z::AbstractArray, x, dec_filters; args=args, scale_offset=args[:scale_offset])
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa, Hx_bounds, Ha_bounds; args=args)
    return full_sequence(models, z0, a0, x, dec_filters; args=args, scale_offset=scale_offset)
end

function model_loss(x, r, dec_filters; args=args)
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa, Hx_bounds, Ha_bounds; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x, dec_filters; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t, dec_filters)
    out = sample_patch(out_small, a1, sampling_grid)
    # Lpatch = Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x, dec_filters;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t, dec_filters)
        out += sample_patch(out_small, a1, sampling_grid)
        # Lpatch += Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss, klqp #+ args[:λpatch] * Lpatch, klqp
end

"output sequence: full recs, local recs, xys (a1), patches_t"
function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    r = gpu(rand(args[:D], args[:π], args[:bsz]))
    dec_filters = get_uncorr_filters(cnn_weights; n_filters=args[:n_filters]) |> gpu
    Hx_bounds, Ha_bounds = get_primary_bounds(args)

    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa, Hx_bounds, Ha_bounds; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x, dec_filters; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t, dec_filters)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t in 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x, dec_filters;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t, dec_filters)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    dec_filters = get_uncorr_filters(cnn_weights; n_filters=args[:n_filters]) |> gpu
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)])
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i], dec_filters)
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
    rs = gpu([rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)])
    L = 0.0f0
    dec_filters = get_uncorr_filters(cnn_weights, n_filters=args[:n_filters]) |> gpu
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(x, rs[i], dec_filters)
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

Zygote.@nograd function get_primary_bounds(args)
    l_enc_za_z = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> z_t+1
    l_fx = get_rnn_θ_sizes(args[:esz], args[:π]) # μ, logvar

    mdec = Chain(HyDense(args[:π], args[:n_filters], args[:bsz], elu),
        flatten,
    )
    l_dec_x = get_param_sizes(mdec)

    Hx_bounds = [l_enc_za_z; l_fx; l_dec_x]

    l_enc_za_a = (args[:π] + args[:asz]) * args[:esz] # encoder (z_t, a_t) -> a_t+1
    l_fa = get_rnn_θ_sizes(args[:esz], args[:ha_sz]) # same size for now
    l_dec_a = args[:ha_sz] * args[:asz] + args[:asz] # decoder z -> a, with bias

    Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

    return Hx_bounds, Ha_bounds
end

## ==== get model etc

function get_RNP_model(args)
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    Hx = Chain(LayerNorm(args[:π]),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, sum(Hx_bounds) + args[:π], bias=false)
    ) |> gpu

    Ha = gpu(Chain(LayerNorm(args[:π]),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, sum(Ha_bounds) + args[:asz]; bias=false)))

    init_hyper!(Hx)
    init_hyper!(Ha)

    Encoder = gpu(let
        enc1 = Chain(x -> reshape(x, args[:img_size]..., args[:img_channels], :),
            Conv((5, 5), args[:img_channels] => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            flatten)
        outsz = Flux.outputsize(enc1,
            (args[:img_size]..., args[:img_channels],
                args[:bsz]))
        Chain(enc1,
            Dense(outsz[1], 64),
            LayerNorm(64, elu),
            Dense(64, 64),
            LayerNorm(64, elu),
            Dense(64, 64),
            LayerNorm(64, elu),
            Dense(64, 64),
            LayerNorm(64, elu),
            Split(Dense(64, args[:π]), Dense(64, args[:π])))
    end)
    return Hx, Ha, Encoder
end

function init_hyper!(Hx; f=2.0f0)
    for p in Flux.params(Hx)
        p ./= f
    end
end

function train_RNP_model(model::Tuple, n_epochs; optimizer=ADAM, args=args, logger=true, save_every=50)
    Hx, Ha, Encoder = model
    ps = Flux.params(Hx, Ha, Encoder)
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    # logger && (lg = new_logger(joinpath(save_folder, alias), args))
    global lg = logger ? new_logger(joinpath(save_folder, alias), args) : nothing
    opt = optimizer(args[:η])
    Ls = []
    for epoch in 1:n_epochs
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6; replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)

        if epoch % save_every == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

function train_RNP_model(args::Dict, n_epochs; optimizer=ADAM, logger=true, save_every=50)
    Hx, Ha, Encoder = model = get_RNP_model(args)
    ps = Flux.params(Hx, Ha, Encoder)
    Hx_bounds, Ha_bounds = get_primary_bounds(args)
    # logger && (lg = new_logger(joinpath(save_folder, alias), args))
    global lg = logger ? new_logger(joinpath(save_folder, alias), args) : nothing
    opt = optimizer(args[:η])
    Ls = []
    for epoch in 1:n_epochs
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6; replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)

        if epoch % save_every == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
    return vcat(Ls...)
end