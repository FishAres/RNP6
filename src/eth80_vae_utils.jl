using DrWatson
include(srcdir("gen_vae_utils.jl"))

"""
! For ETH80 - larger decoder. \n
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

    mdec = Chain(
        HyDense(args[:π], 400, args[:bsz], elu),
        x -> reshape(x, 10, 10, 4, :),
        HyConvTranspose((5, 5), 4 => 32, args[:bsz], relu, stride=1),
        HyConvTranspose((4, 4), 32 => 32, args[:bsz], relu, stride=2, pad=2),
        HyConvTranspose((4, 4), 32 => 3, args[:bsz], relu, stride=2, pad=2)
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
! For ETH80 - larger decoder. \n
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
        HyDense(args[:π], 400, Θvec[4], elu),
        x -> reshape(x, 10, 10, 4, :),
        HyConvTranspose((5, 5), 4 => 32, Θvec[5], relu; stride=1),
        HyConvTranspose((4, 4), 32 => 32, Θvec[6], relu; stride=2, pad=2),
        HyConvTranspose((4, 4), 32 => 3, Θvec[7], relu; stride=2, pad=2)
    )

    z0 = fz.(Θvec[8])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function imview_color(x)
    colorview(RGB, permutedims(batched_adjoint(x), [3, 1, 2]))
end


function plot_rec_color(x, rec, xs::Vector, ind)
    rec_r = reshape(cpu(rec), args[:img_size]..., 3, :)
    x_r = reshape(cpu(x), args[:img_size]..., 3, size(x)[end])
    p_output = plot(imview_color(rec_r[:, :, :, ind]), axis=nothing,)
    p_x = plot(imview_color(x_r[:, :, :, ind]), axis=nothing, size=(20, 20))
    p3 = plot([plot(imview_color(x[:, :, :, ind]), axis=nothing) for x in xs]...)
    return plot(p_output, p_x, p3, layout=(1, 3))
end

function plot_recs(x, inds; args=args)
    full_recs, patches, as, patches_t = get_loop(x)
    patches_r = map(x -> reshape(x, args[:img_size]..., args[:img_channels], size(x)[end]), patches)
    p = [plot_rec_color(full_recs[end], x, patches_r, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end