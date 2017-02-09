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


#####################

# dna

# Source: Statlog / Dna
# Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
# # of classes: 3
# # of data: 2,000 / 1,186 (testing) / 1,400 (tr) / 600 (val)
# # of features: 180
# Files:
# dna.scale
# dna.scale.t (testing)
# dna.scale.tr (tr)
# dna.scale.val (val)
# > C=0.0078125, bias=4
# vote: 0.952
# ctmc: 0.9525
# mean: 0.9525
# prod: 0.9525
features, labels = loaddata("data/dna.scale", 180)
cv(features, labels, C=2.0^-7, bias=2^2)

# satimage

# Source: Statlog / Satimage
# Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
# # of classes: 6
# # of data: 4,435 / 2,000 (testing) / 3,104 (tr) / 1,331 (val)
# # of features: 36
# Files:
# satimage.scale
# satimage.scale.t (testing)
# satimage.scale.tr (tr)
# satimage.scale.val (val)
# > C=8, bias=0.001
# vote: 0.8683201803833145
# ctmc: 0.86967305524239
# mean: 0.8683201803833145
# prod: 0.8683201803833145
features, labels = loaddata("data/satimage.scale", 36)
cv(features, labels, C=2^3, bias=.001)

# letter

# Source: Statlog / Letter
# Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
# # of classes: 26
# # of data: 15,000 / 5,000 (testing) / 10,500 (tr) / 4,500 (val)
# # of features: 16
# Files:
# letter.scale
# letter.scale.t (testing)
# letter.scale.tr (tr)
# letter.scale.val (val)
# > C=8, bias=1
# vote: 0.8450666666666666
# ctmc: 0.8438666666666667
# mean: 0.8445333333333334
# prod: 0.8445333333333334
features, labels = loaddata("data/letter.scale", 16)
cv(features, labels, C=2^3, bias=1)

# shuttle

# Source: Statlog / Shuttle
# Preprocessing: Training data is further separated into two sets, tr and val. [CWH01a]
# # of classes: 7
# # of data: 43,500 / 14,500 (testing) / 30,450 (tr) / 13,050 (val)
# # of features: 9
# Files:
# shuttle.scale
# shuttle.scale.t (testing)
# shuttle.scale.tr (tr)
# shuttle.scale.val (val)
# > C=64, bias=1
# vote: 0.9693103448275862
# ctmc: 0.9695172413793104
# mean: 0.9717701149425287
# prod: 0.9717931034482759
features, labels = loaddata("data/shuttle.scale", 9)
cv(features, labels, C=2^6, bias=1)
