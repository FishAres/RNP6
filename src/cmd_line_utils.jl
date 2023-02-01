using ArgParse

"parse command line arguments"
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_filters"
        help = "number of learned filters used"
        arg_type = Int
        default = 64
        "--device_id"
        help = "GPU device"
        default = 0
    end
    return parse_args(s)
end