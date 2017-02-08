module CTMCEnsemble

export
    build,
    stationdist

"""
    build(preds, nclass=0; weights=nothing)

Build the generator matrix.

# Example

```jldoctest
julia> build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3×3 Array{Float64,2}:
 -0.5   0.5   0.0
  0.5  -1.0   0.5
  0.0   0.5  -0.5
```
"""
function build(preds, nclass=0; weights=nothing)
    nclass == 0 && (nclass = maximum(maximum.(getindex.(preds, 2))))
    weights == nothing && (weights = ones(length(preds)))
    G = zeros(nclass, nclass)
    for ((pred, label), w) in zip(preds, weights)
        G[label, label] .+= w .* assemble(pred)
    end
    G
end

"""
    stationdist(G)

Compute stationary probability distribution given generator ``G``.

# Example

```jldoctest
julia> G = build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])]);

julia> stationdist(G)
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
```
"""
function stationdist(G)
    λ, v = eigs(G, nev=1, which=:LR)
    assert(abs(λ[1]) < 1e-5)
    assert(isreal(v))
    v = vec(real(v))
    v / sum(v)
end

assemble(p::Vector) = repmat(p, 1, length(p)) - eye(length(p))

end # module
