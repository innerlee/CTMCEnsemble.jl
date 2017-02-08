module CTMCEnsemble

export
    average,
    product,
    powermethod,
    svdmethod,
    ctmc

"""
    average(preds, weights=nothing; multiplicity=true)

Compute the degree-weighted average.

# Example

```jldoctest
julia> average([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
```
"""
function average(preds, weights=nothing; multiplicity=true)
    nclass = maximum(maximum.(getindex.(preds, 2)))
    weights == nothing && (weights = ones(length(preds)))
    v = zeros(nclass)
    m = zeros(nclass)
    for ((pred, label), w) in zip(preds, weights)
        v[label] .+= w .* pred
        m[label] .+= w
    end
    if multiplicity
        assert(all(m .> 0))
        v ./= m
    end
    normalize!(v, 1)
end

"""
    product(preds, weights=nothing; multiplicity=true)

Compute the degree-weighted product.

# Example

```jldoctest
julia> product([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
```
"""
function product(preds, weights=nothing; multiplicity=true)
    nclass = maximum(maximum.(getindex.(preds, 2)))
    weights == nothing && (weights = ones(length(preds)))
    v = ones(nclass)
    m = zeros(nclass)
    for ((pred, label), w) in zip(preds, weights)
        v[label] .*= w .* pred
        m[label] .+= w
    end
    if multiplicity
        assert(all(m .> 0))
        v .^= (1 ./ m)
    end
    normalize!(v, 1)
end

assemble(p::Vector) = repmat(p, 1, length(p)) - eye(length(p))

"""
    build(preds, weights=nothing; nclass=0)

Build the generator matrix.

# Example

```jldoctest
julia> CTMCEnsemble.build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3×3 Array{Float64,2}:
 -0.5   0.5   0.0
  0.5  -1.0   0.5
  0.0   0.5  -0.5
```
"""
function build(preds, weights=nothing; nclass=0)
    nclass == 0 && (nclass = maximum(maximum.(getindex.(preds, 2))))
    weights == nothing && (weights = ones(length(preds)))
    G = zeros(nclass, nclass)
    for ((pred, label), w) in zip(preds, weights)
        G[label, label] .+= w .* assemble(pred)
    end
    G
end

"""
    powermethod(preds, weights=nothing; maxiter=16)

Compute stationary distribution by power method.

# Example

```jldoctest
julia> powermethod([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])], maxiter=16)
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
"""
function powermethod(preds, weights=nothing; maxiter=16)
    A = build(preds, weights)
    A .+= eye(A) .* length(preds)
    v = average(preds, weights)

    for i in 1:maxiter
        v = normalize!(A * v, 1)
    end

    v / sum(v)
end

"""
    svdmethod(preds, weights=nothing; maxiter=16)

Compute stationary distribution by svd method.

# Example

```jldoctest
julia> svdmethod([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])], maxiter=16)
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
"""
function svdmethod(preds, weights=nothing;)
    λ, ϕ = eig(build(preds, weights))
    v = real(ϕ[:, indmin(abs(λ))])
    v / sum(v)
end

"""
    stationdist(G)

Compute stationary probability distribution given generator ``G``.

# Example

```jldoctest
julia> G = CTMCEnsemble.build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])]);

julia> CTMCEnsemble.stationdist(G)
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

"""
    ctmc(preds, weights=nothing)

Compute stationary distribution by CTMC method.

# Example

```jldoctest
julia> ctmc([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])], maxiter=16)
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
"""
ctmc(preds, weights=nothing) = stationdist(build(preds, weights))

end # module
