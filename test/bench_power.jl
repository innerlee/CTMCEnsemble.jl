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

# 0=>2951
# 8=>431
# 16=>83
# 24=>18
# 32=>5

bench(1)
bench(2)
bench(3)
bench(4)
bench(5)

println("love bufan")
