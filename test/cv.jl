using LIBLINEAR
using CTMCEnsemble

function vote(p...)
    m = countmap([p...])
    collect(keys(m))[indmax(values(m))]
end

vote_all(preds) = vote.((getindex.(preds,1))...)

onehot(i, n, eps=0) = setindex!(ones(n) * eps / (n - 1), 1 - eps, i)

function pred_all(preds, method)
    results = Int[]
    n = size(preds[1][1], 1)
    for i = 1:n
        p = []
        for (pr, ran) in preds
            push!(p, (onehot(findfirst(ran, pr[i]), length(ran), 1e-5), ran))
        end
        push!(results, indmax(method(p)))
    end
    results
end

function cv(features, labels; nfold=10, C=1, bias=1)
    srand(0) # fix seed
    reorder = shuffle(1:length(labels))
    d = features[:, reorder]
    l = labels[reorder]

    results = []
    voteacc = 0
    ctmcacc = 0
    meanacc = 0
    prodacc = 0
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
        ctmcacc += sum(pred_all(preds, ctmc) .== testlabel)
        meanacc += sum(pred_all(preds, average) .== testlabel)
        prodacc += sum(pred_all(preds, product) .== testlabel)
    end

    println("vote: ", voteacc / length(l))
    println("ctmc: ", ctmcacc / length(l))
    println("mean: ", meanacc / length(l))
    println("prod: ", prodacc / length(l))
end
