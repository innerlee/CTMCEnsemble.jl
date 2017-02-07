using CTMCEnsemble
using Base.Test

# write your own tests here

@testset "docs" begin
    docs = readstring(`$(Base.julia_cmd()) -e '
        cd("../docs");include("../docs/make.jl")'`)
    contains(docs, "!!") && println(docs)
    @test 1 == 1
end
