function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            return model_loss(x, y)
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data)
    L = 0.0f0
    for (x, y) in test_data
        L += model_loss(x, y)
    end
    return L / length(test_data)
end
