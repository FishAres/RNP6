using DrWatson
using TensorBoardLogger
using Logging

"create tensorboard logger"
savename_(args) = savename(args; allowedtypes=(Real, String, Symbol, Function))

function new_logger(path, args)
    return TBLogger("tensorboard_logs/$(path)/$(savename(args))"; min_level=Logging.Info)
end

function get_save_dir(save_folder, alias)
    save_dir = joinpath("saved_models", save_folder, alias)
    try
        mkpath(save_dir)
    catch
        println("directories already there :)")
    end
    return save_dir
end
