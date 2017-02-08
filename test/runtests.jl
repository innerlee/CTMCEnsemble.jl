using CTMCEnsemble
using Base.Test

@testset "example" begin
    preds = [([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])]
    G = CTMCEnsemble.build(preds)
    @test CTMCEnsemble.stationdist(G) ≈ [1/3, 1/3, 1/3]

    @test [1/3, 1/3, 1/3] ≈ average(preds)
    @test [1/4, 1/2, 1/4] ≈ average(preds, multiplicity=false)
    @test [1/3, 1/3, 1/3] ≈ product(preds)
    @test [2/5, 1/5, 2/5] ≈ product(preds, multiplicity=false)
    @test [1/3, 1/3, 1/3] ≈ powermethod(preds)
    @test [1/3, 1/3, 1/3] ≈ svdmethod(preds)
    @test [1/3, 1/3, 1/3] ≈ ctmc(preds)
end

@testset "single" begin
    # single
    for i = 1:5
        d = 1000
        p = normalize!(rand(d), 1)
        preds = [(p, 1:d)]

        @test p ≈ average(preds)
        @test p ≈ average(preds, multiplicity=false)
        @test p ≈ product(preds)
        @test p ≈ product(preds, multiplicity=false)
        @test p ≈ powermethod(preds)
        @test p ≈ svdmethod(preds)
        @test p ≈ ctmc(preds)
    end
end

@testset "multiple" begin
    newv() = normalize!(rand(300), 1)

    for i = 1:5
        d = 600
        preds = [
            (newv(),     1:300),
            (newv(),   201:500),
            (newv(), ((401:700) - 1) % 600 + 1),
        ]

        v_powermethod = powermethod(preds, maxiter=64)
        v_svdmethod   = svdmethod(preds)
        v_ctmc        = ctmc(preds)

        @test v_svdmethod ≈ v_ctmc
        @test_approx_eq_eps v_svdmethod v_powermethod 1/d
    end
end

@testset "love bufan" begin
    @test 1 == 1
end
