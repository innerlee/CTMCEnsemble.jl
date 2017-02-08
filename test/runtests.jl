using CTMCEnsemble
using Base.Test

@testset "example" begin
    G = build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
    @test stationdist(G) ≈ [1/3, 1/3, 1/3]
end

@testset "random cases" begin
    # single
    for i = 1:10
        d = 1000
        p = normalize!(rand(d), 1)
        G = build([(p, 1:d)])
        @test stationdist(G) ≈ p
    end

    # multiple
    for i = 1:10
        d = 600
        newv() = normalize!(rand(300), 1)
        G = build([
            (newv(),     1:300),
            (newv(),   201:500),
            (newv(), ((401:700) - 1) % 600 + 1),
        ])
        λ, ϕ = eig(G)
        v = real(ϕ[:, indmin(abs(λ))])
        v = v / sum(v)

        @test stationdist(G) ≈ v
    end

end

@testset "love bufan" begin
    @test 1 == 1
end
