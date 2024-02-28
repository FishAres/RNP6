using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Distributions
using ProgressMeter
using ProgressMeter: Progress
using Plots

include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("models.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))

include(srcdir("utils.jl"))
## ======

sample_z(μ, logvar, r) = μ + r .* (exp.(logvar))

kl_loss(μ, logvar) = sum(@. (exp(logvar) + μ^2 - logvar - 1.0f0))

"""
Get indices at which to slice hypernet outputs
#Arguments
- `args::Dict`: script arguments
"""
function get_primary_bounds(args)
    # == State network
    mEnc_za_z = Chain( # dummy network to get parameter sizes from
        HyDense(args[:π] + args[:asz], args[:layer_sz], args[:bsz], elu),
        flatten,
        HyDense(args[:layer_sz], args[:esz], args[:bsz], elu),
        flatten,
    )
    l_enc_za_z = get_param_sizes(mEnc_za_z)
    l_fx = get_rnn_θ_sizes(args[:esz], args[:π])

    mdec = Chain( # dummy network
        HyDense(args[:π], args[:layer_sz], args[:bsz], elu),
        flatten,
        HyDense(args[:layer_sz], 784, args[:bsz], relu),
        flatten,
    )
    l_dec_x = get_param_sizes(mdec)

    Hstate_bounds = [l_enc_za_z...; l_fx; l_dec_x...]

    # == Policy network
    mEnc_za_a = Chain( # dummy network
        HyDense(args[:π] + args[:asz], args[:layer_sz], args[:bsz], elu),
        flatten,
        HyDense(args[:layer_sz], args[:esz], args[:bsz], elu),
        flatten,
    )
    l_enc_za_a = get_param_sizes(mEnc_za_a)
    l_fa = get_rnn_θ_sizes(args[:esz], args[:π]) # same size for now
    l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

    Hpolicy_bounds = [l_enc_za_a...; l_fa; l_dec_a]
    return Hstate_bounds, Hpolicy_bounds
end


"""
Generate the "state" models of an RNP module, given the output of H_state
# Arguments
- `θs::Array`: output of state hypernetwork
- `Hstate_bounds::Vector`: vector of indices at which to slice the parameter vector,
 corresponding to individual networks in the RNP module
- `fz::function`: The activation function for the state RNN
"""
function get_fstate_models(θs, Hstate_bounds; args=args, fz=args[:f_z])
    @assert size(θs, 1) == sum([Hstate_bounds...; args[:π]])
    inds = Zygote.ignore() do # pad parameter slicing offsets
        return [0; cumsum([Hstate_bounds...; args[:π]])]
    end
    Θvec = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_z = Chain(
        HyDense(args[:π] + args[:asz], args[:layer_sz], Θvec[1], elu),
        flatten,
        HyDense(args[:layer_sz], args[:esz], Θvec[2], elu),
        flatten
    )

    f_state = ps_to_RN(get_rn_θs(Θvec[3], args[:esz], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(
        HyDense(args[:π], args[:layer_sz], Θvec[4], elu),
        flatten,
        HyDense(args[:layer_sz], 784, Θvec[5], relu6),
        flatten)

    z0 = fz.(Θvec[6])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

"""
Generate the "state" models of an RNP module, given the output of H_policy
# Arguments
- `θs::Array`: output of policy hypernetwork
- `Hpolicy_bounds::Vector`: vector of indices at which to slice the parameter vector,
 corresponding to individual networks in the RNP module
"""
function get_fpolicy_models(θs, Hpolicy_bounds; args=args)
    @assert size(θs, 1) == sum([Hpolicy_bounds...; args[:asz]])
    inds = Zygote.ignore() do # pad parameter slicing offsets
        return [0; cumsum([Hpolicy_bounds...; args[:asz]])]
    end
    Θvec = [θs[(inds[i]+1):inds[i+1], :] for i in 1:(length(inds)-1)]

    Enc_za_a = Chain(
        HyDense(args[:π] + args[:asz], args[:layer_sz], Θvec[1], elu),
        HyDense(args[:layer_sz], args[:esz], Θvec[2], elu),
        flatten
    )
    f_policy = ps_to_RN(get_rn_θs(Θvec[3], args[:esz], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θvec[4], sin), flatten)
    a0 = sin.(Θvec[5])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

"""
Get full RNP module from hypernet outputs
#Arguments
- `θs_state::Array`: output of H_state hypernetwork
- `θs_policy::Array`: output of H_policy hypernetwork
`Hstate_bounds::Vector[Int]` and `Hpolicy_bounds::Vector[Int]`
must be in scope (to avoid propagating their value everywhere) 
"""
function get_models(θs_state, θs_policy; args=args)
    (Enc_za_z, f_state, Dec_z_x̂), z0 = get_fstate_models(θs_state, Hstate_bounds; args=args)
    (Enc_za_a, f_policy, Dec_z_a), a0 = get_fpolicy_models(θs_policy, Hpolicy_bounds; args=args)
    # pack models into tuple 
    models = f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a
    return models, z0, a0
end

"""
Get full RNP module from ``z`` vector at level ``k``
"""
function get_models(z; args=args)
    θs_state = H_state(z)
    θs_policy = H_policy(z)
    # yay multiple dispatch
    models, z0, a0 = get_models(θs_state, θs_policy; args=args)
    return models, z0, a0
end

"""
Single RNP pass for level ``k``
# Arguments
- `z_prev::Array`: the ``zᵏ`` vector
- `a_prev::Array`: the ``aᵏ`` vector
- `models::Tuple`: tuple of models comprising the RNP module at level ``k``
- `x::Array`: the input image patch at level ``k``
- `scale_offset::Float32`: offset added to the affine transformation array
when scaling a patch. Defaults to `args[:scale_offset]`

"""
function forward_pass(z_prev, a_prev, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z_a_concat = vcat(z_prev, a_prev)
    f_state_input = Enc_za_z(z_a_concat)
    f_policy_input = Enc_za_a(z_a_concat)
    z_new = f_state(f_state_input) # new z, can be passed into the hypernets
    a_new = Dec_z_a(f_policy(f_policy_input)) # new affine parameters

    x̂ = Dec_z_x̂(z_new) # image patch generated from z_new
    # section of image (x) specified by affine parameters (i.e. "zoomed in")
    patch_t = flatten(zoom_in2d(x, a_new, sampling_grid; scale_offset=scale_offset))

    return z_new, a_new, x̂, patch_t
end

"""
Generate a full sequence at level ``k`` using existing RNP module
#Arguments
- `models::Tuple`: tuple of models comprising the RNP module at level ``k``
- `z0::Array`: a ``z`` vector to initialize the sequence, i.e. ``zᵏ₀``
- `a0::Array`: an affine transform vector to initialize the sequence, i.e. ``aᵏ₀``
- `x::Array`: the image patch at level ``k``; used to compute section of x specified
by affine parameters at each ``a_t``
"""
function full_sequence(models::Tuple, z0, a0, x; args=args,
    scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z_t, a_t, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    # initialize the output patch for the sequence
    output = sample_patch(x̂, a_t, sampling_grid)
    for t in 2:args[:seqlen]
        z_t, a_t, x̂, patch_t = forward_pass(z_t, a_t, models, x; scale_offset=scale_offset)
        output += sample_patch(x̂, a_t, sampling_grid)
    end
    return output
end

"""
Generate a full sequence at level ``k`` from vector ``z`` at ``k+1``
Assumes hypernets H_state and H_policy are in scope
"""
function full_sequence(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = H_state(z)
    θsa = H_policy(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

"""
Calculate the model loss for target image ``x``
#Arguments
- `x::Array`: Target image
- `r::Array`: random vector used for reparameterization
Assumes hypernetworks H_state and H_policy are in scope
"""
function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θs_state = H_state(z)
    θs_policy = H_policy(z)
    models, z0, a0 = get_models(θs_state, θs_policy; args=args)
    z_t, a_t, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z_t, patch_t)
    output = sample_patch(out_small, a_t, sampling_grid)
    # Patch loss - optional
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t in 2:args[:seqlen]
        z_t, a_t, x̂, patch_t = forward_pass(z_t, a_t, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z_t, patch_t)
        output += sample_patch(out_small, a_t, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(output), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

"""
Dirty function to reduce number of pushes in `get_loop``
"""
Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end

"""
Generate output sequence: full recs, local recs, affine transforms (as), image patches
"""
function get_loop(x; args=args)
    output_list = patches, recs, as, patches_t = [], [], [], [], []
    r = dev(randn(Float32, args[:π], args[:bsz]))
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θs_state = H_state(z)
    θs_policy = H_policy(z)
    models, z0, a0 = get_models(θs_state, θs_policy; args=args)
    z_t, a_t, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z_t, patch_t)
    output = sample_patch(out_small, a_t, sampling_grid)
    push_to_arrays!((output, out_small, a_t, patch_t), output_list)
    for t in 2:args[:seqlen]
        z_t, a_t, x̂, patch_t = forward_pass(z_t, a_t, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z_t, patch_t)
        output += sample_patch(out_small, a_t, sampling_grid)
        push_to_arrays!((output, out_small, a_t, patch_t), output_list)
    end
    return output_list
end

"Generate 2-level samples from ``z²`` vector `z`"
function sample_(z, x; args=args)
    θs_state = H_state(z)
    θs_policy = H_policy(z)
    models, z0, a0 = get_models(θs_state, θs_policy; args=args)
    z_t, a_t, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z_t, patch_t)
    output = sample_patch(out_small, a_t, sampling_grid)
    for t in 2:args[:seqlen]
        z_t, a_t, x̂, patch_t = forward_pass(z_t, a_t, models, x;
            scale_offset=args[:scale_offset])
        out_small = full_sequence(z_t, patch_t)
        output += sample_patch(out_small, a_t, sampling_grid)
    end
    return output |> cpu
end


"""
Train an RNP for one epoch! 
# Arguments
- `opt::Optimiser`: Your optimizer
- `ps::Flux.params`: (Implicit) model parameters 
- `train_loader::DataLoader`: Data loader containing your train data
- `logger::TensorBoardLogger`: Optional TensorBoard logger
- `clip_grads::Bool`: Whether or not to clip gradients before updating parameters

Note: Device `dev` (`const`) must be in global scope
"""
function train_epoch(opt, ps, train_loader; args=args, epoch=1, logger=nothing, clip_grads=false)
    progress_tracker = Progress(length(train_loader), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_loader))
    # initial z's drawn from N(0,1)
    rs = dev([randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_loader)])
    for (i, x) in enumerate(train_loader)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                return log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            return full_loss + args[:λ] * (norm(Flux.params(H_state)) + norm(Flux.params(H_policy)))
        end
        clip_grads && foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

"""
Test loss for RNP

Note: Device `dev` (`const`) must be in global scope
"""
function test_model(test_loader; args=args)
    rs = dev([randn(Float32, args[:π], args[:bsz]) for _ in 1:length(test_loader)])
    L = 0.0f0 # initialize test loss to 0
    for (i, x) in enumerate(test_loader)
        rec_loss, klqp = model_loss(x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_loader)
end

function train_model(ps, train_loader, test_loader, save_dir; args=args, n_epochs=10, logger=nothing, clip_grads=false, sample_every=5, save_every=10, inc_lpatch=false)
    has_logger = logger !== nothing
    opt = ADAM(args[:lr])
    losses = []
    for epoch in 1:n_epochs

        inc_lpatch & (epoch % 5 == 0) && begin # increment patch loss
            args[:λpatch] = max(args[:λpatch] + 1.0f-3, 1.0f-2)
            has_logger && log_value(logger, "λpatch", args[:λpatch])
        end

        ls = train_epoch(opt, ps, train_loader; epoch=epoch, logger=logger, clip_grads=clip_grads)

        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        has_logger && log_image(logger, "recs_$(epoch)", p)

        if epoch % sample_every == 0
            z = randn(Float32, args[:π], args[:bsz]) |> gpu
            out = sample_(z, x)
            psamp = stack_ims(out) |> plot_digit
            has_logger && log_image(lg, "sampling_$(epoch)", psamp)
            display(psamp)
        end

        Ltest = test_model(test_loader)
        has_logger && log_value(logger, "test loss", Lval)
        @info "Test loss: $Ltest"

        epoch % save_every == 0 && begin
            save_model((H_state, H_policy, Encoder), joinpath("saved_models/" * save_dir, savename(args) * "_$(epoch)eps"))
        end
        push!(losses, ls)
    end
    return vcat(losses...)
end


## ==== plotting =====

"""
Plot the reconstruction next to the input image
# Arguments
- `reconstruction::Array`: the reconstruction/output image of an RNP
- `x::Array`: the target image
- `batch_ind::Int`: the element in the batch of `x` and `reconstruction` to plot
"""
function plot_rec(reconstruction, x, batch_ind; kwargs...)
    out_reshaped = reshape(cpu(reconstruction), args[:img_size]..., size(out)[end])
    x_ = reshape(cpu(x), args[:img_size]..., size(x)[end])
    p_x = plot_digit(x_[:, :, batch_ind])
    p_output = plot_digit(out_reshaped[:, :, batch_ind])
    return plot(p_x, p_output, kwargs...)
end

"""
Plot reconstructions and generated patches next to the input image
# Arguments
- `reconstruction::Array`: the reconstruction/output image of an RNP
- `x::Array`: the target image
- `patches::Vector[Array]`:: the patches at k=1
- `batch_ind::Int`: the element in the batch of `x` and `reconstruction` to plot
"""
function plot_rec_seq(reconstruction, x, patches, batch_ind; args=args)
    out_ = reshape(cpu(reconstruction), args[:img_size]..., :)
    x_ = reshape(cpu(x), args[:img_size]..., :)
    p_x = plot_digit(x_[:, :, batch_ind])
    p_output = plot_digit(out_[:, :, batch_ind])
    p_patch = plot([plot_digit(patch[:, :, 1, batch_ind]; boundc=false) for patch in patches]...)
    return plot(p_x, p_output, p_patch; layout=(1, 3))
end

"""
Plot a vector of `length(batch_inds)` rows containing reconstructions and input images
# Arguments
- `x::Array`: input image
- `batch_inds:Vector[Int]`: batch elements to plot
- `plot_seq::Bool`: whether or not to plot generated patches on each row
"""
function plot_recs(x, batch_inds; plot_seq=true, args=args)
    full_recs, patches, _, _ = get_loop(x)
    p = if plot_seq # also plot patch sequence
        let
            # reshape patches
            patches_r = map(x -> reshape(x, args[:img_size]..., 1, size(x)[end]), patches)
            [plot_rec_seq(full_recs[end], x, patches_r, batch_ind) for batch_ind in batch_inds]
        end
    else
        [plot_rec(full_recs[end], x, batch_ind) for batch_ind in batch_inds]
    end

    return plot(p...; layout=(length(batch_inds), 1), size=(600, 800))
end

"""
Hacky function to get a random batch from a data loader
#Arguments
- `loader::DataLoader`: your data loader
"""
function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_batch = for (i, x) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    return x_batch
end
