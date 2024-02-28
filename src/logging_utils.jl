using DrWatson
using TensorBoardLogger
using Logging

"Extend `savename` to allow logging functions"
savename_(args) = savename(args; allowedtypes=(Real, String, Symbol, Function))

"Create tensorboard logger"
function new_logger(path, args)
    return TBLogger("tensorboard_logs/$(path)/$(savename(args))"; min_level=Logging.Info)
end

"""
Convenience function to create a directory wherein thou might save models
    with a given alias. Useful when you're training with different conditions
"""
function get_save_dir(save_folder, alias)
    save_dir = joinpath("saved_models", save_folder, alias)
    try
        mkpath(save_dir)
    catch
        println("directory already there :)")
    end
    return save_dir
end
