using CTMCEnsemble
using Base.Test

@testset "example" begin
    G = build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
    @test stationdist(G) ≈ [1/3, 1/3, 1/3]
end

@testset "love bufan" begin
    @test 1 == 1
end
