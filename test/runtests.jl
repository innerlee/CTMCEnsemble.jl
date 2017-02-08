using CTMCEnsemble
using Base.Test

@testset "example" begin
    G = build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
    @test stationdist(G) ≈ [1/3, 1/3, 1/3]
end

@testset "random cases" begin
    for i = 1:100
        d = 1000
        p = normalize!(rand(d), 1)
        G = build([(p, 1:d)])
        @test stationdist(G) ≈ p
    end
end

@testset "love bufan" begin
    @test 1 == 1
end