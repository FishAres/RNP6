using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using Images
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## =======

device!(0)

## ======

datapath = "/gscratch/rao/aresf/Code/datasets/ETH-80/"

function load_eth_ims(datapath; newres=(50, 50))
    img_files, map_files = [], []
    @inbounds for (root, dirs, files) in walkdir(datapath)
        for file in files
            if endswith(file, "-map.png")
                map = channelview(imresize(load(joinpath(root, file)), newres))
                push!(map_files, Float32.(map))
            elseif endswith(file, ".png")
                img = channelview(imresize(load(joinpath(root, file)), newres))
                push!(img_files, Float32.(img))
            end
        end
    end
    return img_files, map_files
end

# save(datadir("exp_pro", "eth80_imgs_maps.jld2"), Dict("imgs" => img_files, "maps" => map_files))

## =====

function segment_eth80(img_files, map_files)
    @assert length(img_files) == length(map_files)
    filenum = length(img_files)
    array_size = size(img_files[1])
    segmented_ims = zeros(Float32, array_size[[2, 3, 1]]..., filenum)
    Threads.@threads for i in 1:filenum
        imfile = img_files[i]
        imfile[map_files[i].==0.0f0] .= 0.0f0
        segmented_ims[:, :, :, i] = permutedims(imfile, [2, 3, 1])
    end
    return segmented_ims
end

## run
img_files, map_files = load_eth_ims(datapath)

@time seg_ims = segment_eth80(img_files, map_files)
save(datadir("exp_pro", "eth80_segmented.jld2"), Dict("seg_ims" => seg_ims))
