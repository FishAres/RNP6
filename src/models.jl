"""
Get hypernetworks and encoder.
Note: `Hstate_bounds` and `Hpolicy_bounds` must be (`const`) in scope
"""
function generate_hypernets(args)

    H_state = Chain(LayerNorm(args[:π]),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, sum(Hstate_bounds) + args[:π], bias=false),
    )

    H_policy = Chain(LayerNorm(args[:π]),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, sum(Hpolicy_bounds) + args[:asz]; bias=false)
    )

    Encoder = let
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
            BasicBlock(32 => 32, +),
            flatten)
        outsz = Flux.outputsize(enc1,
            (args[:img_size]..., args[:img_channels],
                args[:bsz]))
        Chain(enc1,
            Dense(outsz[1], 128),
            LayerNorm(128, elu),
            Dense(128, 128),
            LayerNorm(128, elu),
            Dense(128, 128),
            LayerNorm(128, elu),
            Dense(128, 128),
            LayerNorm(128, elu),
            Split(Dense(128, args[:π]), Dense(128, args[:π])))
    end
    return H_state, H_policy, Encoder
end

