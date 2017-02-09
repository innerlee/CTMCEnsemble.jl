include("cv.jl")

# glass
# Source: UCI / Glass Identification
# # of classes: 6
# # of data: 214
# # of features: 9
# Files:
# glass.scale
features, labels = loaddata("data/glass.scale", 9)

cv(features, labels, C=2^5, bias=1)
