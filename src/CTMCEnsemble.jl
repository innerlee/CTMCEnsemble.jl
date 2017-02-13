module CTMCEnsemble

export
    average,
    product,
    powermethod,
    svdmethod,
    ctmc

"""
    average(preds, weights=nothing; multiplicity=true)

Compute the average. `multiplicity` for arithmetic mean based on mulitpiliciity.

# Example

```jldoctest
julia> average([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333

julia> average([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])], multiplicity=false)
3-element Array{Float64,1}:
 0.25
 0.5
 0.25
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

Compute the product. `multiplicity` for geometric mean based on mulitpiliciity.

# Example

```jldoctest
julia> product([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333

julia> product([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])], multiplicity=false)
3-element Array{Float64,1}:
 0.4
 0.2
 0.4
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

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> powermethod([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.333333  0.0244257  0.0217406  0.230769
 0.333333  0.0976007  0.195655   0.538462
 0.333333  0.877974   0.782604   0.230769
```
"""
function powermethod(preds, weights=nothing; maxiter=24)
    ndims(preds[1][1]) == 1 &&
        return _powermethod(preds, weights)
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, _powermethod(map(x -> (x[1][:, i], x[2]), preds), weights, maxiter=maxiter))
    end
    hcat(ans...)
end

function _powermethod(preds, weights=nothing; maxiter=24)
    weights == nothing && (weights = ones(length(preds)))
    A = build(preds, weights)
    A .+= eye(A) .* sum(weights)
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
julia> svdmethod([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> svdmethod([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.333333  0.0243902  0.0217391  0.230769
 0.333333  0.097561   0.195652   0.538462
 0.333333  0.878049   0.782609   0.230769
```
"""
function svdmethod(preds, weights=nothing)
    ndims(preds[1][1]) == 1 &&
        return _svdmethod(preds, weights)
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, _svdmethod(map(x -> (x[1][:, i], x[2]), preds), weights))
    end
    hcat(ans...)
end

function _svdmethod(preds, weights=nothing;)
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
julia> ctmc([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> ctmc([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.333333  0.0243902  0.0217391  0.230769
 0.333333  0.097561   0.195652   0.538462
 0.333333  0.878049   0.782609   0.230769
```
"""
function ctmc(preds, weights=nothing)
    ndims(preds[1][1]) == 1 &&
        return stationdist(build(preds, weights))
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, stationdist(build(map(x -> (x[1][:, i], x[2]), preds), weights)))
    end
    hcat(ans...)
end

end # module
