using CTMCEnsemble
using Base.Test
using StatsBase

"""
    bench(sparsity=3, num=10000)

Find a good `maxiter` for power method.
`sparsity` controls generated distribution.
`num` is repeat number.
"""
function bench(sparsity=3, num=10000)
    newv() = normalize!(exp.(sparsity * randn(300)), 1)
    println("> sparsity=", sparsity, ", repeat=", num)
    stats = []
    for i = 1:num
        d = 600
        preds = [
            (newv(),     1:300),
            (newv(),   201:500),
            (newv(), ((401:700) - 1) % 600 + 1),
        ]
        i % 1000 == 0 && print(".")
        for maxiter = 0:8:32
            v_powermethod = powermethod(preds, maxiter=maxiter)
            v_ctmc        = ctmc(preds)
            diff = v_powermethod - v_ctmc
            same = indmax(v_powermethod) == indmax(v_ctmc)
            same || push!(stats, maxiter)
            # println(same ? "[O]" : "[X]"),
            #         "iter: ", maxiter,
            #         ", $d distinf = ", d * norm(diff, Inf),
            #         ", $d dist2 = ", d * norm(diff))
        end

    end
    m = collect(countmap(stats))
    print()
    println(join(join.(m[sortperm(getindex.(m, 1))], ["=>"]), "\n"))
    m[sortperm(getindex.(m, 1))]
end

bench(1)
bench(2)
bench(3)
bench(4)
bench(5)

# > sparsity=1, repeat=10000
# ..........0=>820
# 8=>146
# 16=>30
# 24=>7
# 32=>3
# > sparsity=2, repeat=10000
# ..........0=>1486
# 8=>240
# 16=>52
# 24=>7
# > sparsity=3, repeat=10000
# ..........0=>2180
# 8=>320
# 16=>86
# 24=>30
# 32=>10
# > sparsity=4, repeat=10000
# ..........0=>2465
# 8=>349
# 16=>115
# 24=>57
# 32=>33
# > sparsity=5, repeat=10000
# ..........0=>2567
# 8=>431
# 16=>187
# 24=>107
# 32=>82

println("love bufan")
