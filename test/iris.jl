using RDatasets

include("cv.jl")

iris = dataset("datasets", "iris")

features = Array(iris[:, 1:4])'
ma = maximum(features, 2)
mi = minimum(features, 2)
features = -1 + 2 * (features .- mi) ./ (ma - mi)

# map labels to ints
classes  = unique(iris[:Species])
label2num = Dict()
for (i, l) in enumerate(classes)
    label2num[l] = i
end

labels = Vector{Int}(getindex.([label2num], iris[:Species]))

cv(features, labels, C=2^5, bias=1)
