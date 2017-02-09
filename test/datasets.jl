include("cv.jl")

# glass

# Source: UCI / Glass Identification
# # of classes: 6
# # of data: 214
# # of features: 9
# Files:
# glass.scale
# > C=32, bias=1
# vote: 0.6261682242990654
# ctmc: 0.6355140186915887
# mean: 0.6355140186915887
# prod: 0.6261682242990654
features, labels = loaddata("data/glass.scale", 9)
cv(features, labels, C=2^5, bias=1)

# iris

# Source: UCI / Iris Plant
# # of classes: 3
# # of data: 150
# # of features: 4
# Files:
# iris.scale
# > C=32, bias=1
# vote: 0.98
# ctmc: 0.98
# mean: 0.98
# prod: 0.98
features, labels = loaddata("data/iris.scale", 4)
cv(features, labels, C=2^5, bias=1)

# segment

# Source: Statlog / Segment
# # of classes: 7
# # of data: 2,310
# # of features: 19
# Files:
# segment.scale
# > C=16, bias=1
# vote: 0.9575757575757575
# ctmc: 0.9575757575757575
# mean: 0.9571428571428572
# prod: 0.9571428571428572
features, labels = loaddata("data/segment.scale", 19)
cv(features, labels, C=2^5, bias=1)

# vehicle

# Source: Statlog / Vehicle
# # of classes: 4
# # of data: 846
# # of features: 18
# Files:
# vehicle.scale
# > C=32, bias=1
# vote: 0.8226950354609929
# ctmc: 0.8250591016548463
# mean: 0.8191489361702128
# prod: 0.8191489361702128
features, labels = loaddata("data/vehicle.scale", 18)
cv(features, labels, C=2^5, bias=1)

# vowel

# Source: UCI / Vowel
# Preprocessing: First 528 instances are used as training and the remaining instances are for testing. Scaling training data first and adjust testing data accordingly.
# # of classes: 11
# # of data: 528 / 462 (testing)
# # of features: 10
# Files:
# vowel
# vowel.t (testing)
# vowel.scale (scaled to [-1,1])
# vowel.scale.t (testing)
# > C=8, bias=1
# vote: 0.7537878787878788
# ctmc: 0.7424242424242424
# mean: 0.7443181818181818
# prod: 0.7443181818181818
features, labels = loaddata("data/vowel.scale", 10)
cv(features, labels, C=2^5, bias=1)

# wine

# Source: UCI / Wine Recognition
# # of classes: 3
# # of data: 178
# # of features: 13
# Files:
# wine.scale
# > C=0.0625, bias=1
# vote: 0.9775280898876404
# ctmc: 0.9775280898876404
# mean: 0.9775280898876404
# prod: 0.9775280898876404
features, labels = loaddata("data/wine.scale", 13)
cv(features, labels, C=2.0^-2, bias=1)
