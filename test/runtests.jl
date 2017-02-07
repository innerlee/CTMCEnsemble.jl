using CTMCEnsemble
using Base.Test

# write your own tests here

@testset "love bufan" begin
    readstring(`$(Base.julia_cmd()) -e 'cd("../docs");include("../docs/make.jl")'`)
    @test 1 == 1
end
