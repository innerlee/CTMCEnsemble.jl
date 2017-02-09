using RDatasets
using LIBLINEAR
using CTMCEnsemble

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

function vote(p...)
    m = countmap([p...])
    collect(keys(m))[indmax(values(m))]
end

vote_all(preds) = vote.((getindex.(preds,1))...)

onehot(i, n, eps=0) = setindex!(ones(n) * eps / (n - 1), 1 - eps, i)

function ctmc_all(preds)
    results = Int[]
    n = size(preds[1][1], 1)
    for i = 1:n
        p = []
        for (pr, ran) in preds
            push!(p, (onehot(findfirst(ran, pr[i]), length(ran), 1e-5), ran))
        end
        push!(results, indmax(ctmc(p)))
    end
    results
end

preds = 0
# 10-cv
C = 1
bias = 1
srand(0) # fix seed
nfold = 10
reorder = shuffle(1:length(labels))
d = features[:, reorder]
l = labels[reorder]

results = []
voteacc = 0
ctmcacc = 0
for f = 1:nfold
    trainmask  = trues(length(l))
    trainmask[f:nfold:end] = false
    traindata  = d[:, trainmask]
    trainlabel = l[trainmask]
    testdata   = d[:, !trainmask]
    testlabel  = l[!trainmask]

    preds = []
    for i=1:length(classes), j=i+1:length(classes)
        localcls = [i, j]
        trainmask = in.(trainlabel, [localcls])
        # testmask = in.(testlabel, [localcls])
        m = linear_train(trainlabel[trainmask], traindata[:, trainmask], C=C, bias=bias)
        p, _ = linear_predict(m, testdata)
        push!(preds, (p, localcls))
    end

    voteacc += sum(vote_all(preds) .== testlabel)
    ctmcacc += sum(ctmc_all(preds) .== testlabel)
end

println(voteacc / length(l))
println(ctmcacc / length(l))
