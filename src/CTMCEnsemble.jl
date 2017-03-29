module CTMCEnsemble

export
    average,
    product,
    vote,
    sink,
    powermethod,
    svdmethod,
    ctmc,
    top1,
    top5,
    softmax!

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

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> average([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.333333  0.129032  0.0689655  0.230769
 0.333333  0.290323  0.37931    0.538462
 0.333333  0.580645  0.551724   0.230769

julia> average([(A, [1, 2]), (B, [2, 3])], multiplicity=false)
3×4 Array{Float64,2}:
 0.25  0.1   0.05  0.15
 0.5   0.45  0.55  0.7
 0.25  0.45  0.4   0.15
```
"""
function average(preds, weights=nothing; multiplicity=true)
    ndims(preds[1][1]) == 1 &&
        return _average(preds, weights, multiplicity=multiplicity)
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, _average(map(x -> (x[1][:, i], x[2]), preds), weights, multiplicity=multiplicity))
    end
    hcat(ans...)
end

function _average(preds, weights=nothing; multiplicity=true)
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

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> product([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.333333  0.14463   0.0755136  0.230769
 0.333333  0.204537  0.320377   0.538462
 0.333333  0.650833  0.604109   0.230769

julia> product([(A, [1, 2]), (B, [2, 3])], multiplicity=false)
3×4 Array{Float64,2}:
 0.4  0.169492   0.0925926  0.275229
 0.2  0.0677966  0.166667   0.449541
 0.4  0.762712   0.740741   0.275229
```
"""
function product(preds, weights=nothing; multiplicity=true)
    ndims(preds[1][1]) == 1 &&
        return _product(preds, weights, multiplicity=multiplicity)
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, _product(map(x -> (x[1][:, i], x[2]), preds), weights, multiplicity=multiplicity))
    end
    hcat(ans...)
end

function _product(preds, weights=nothing; multiplicity=true)
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

"""
    vote(preds, weights=nothing)

Compute the vote.

# Example

```jldoctest
julia> vote([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 0.5
 0.5
 0.0

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> vote([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 0.5  0.0  0.0  0.0
 0.5  0.5  0.5  1.0
 0.0  0.5  0.5  0.0
```
"""
function vote(preds, weights=nothing; multiplicity=true)
    ndims(preds[1][1]) == 1 &&
        return _vote(preds, weights, multiplicity=multiplicity)
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, _vote(map(x -> (x[1][:, i], x[2]), preds), weights, multiplicity=multiplicity))
    end
    hcat(ans...)
end

function _vote(preds, weights=nothing; multiplicity=true)
    nclass = maximum(maximum.(getindex.(preds, 2)))
    weights == nothing && (weights = ones(length(preds)))
    v = zeros(nclass)
    for ((pred, label), w) in zip(preds, weights)
        i = label[indmax(pred)]
        v[i] .+= w
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
    svdmethod(preds, weights=nothing)

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
    v = real(ϕ[:, indmin(abs.(λ))])
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
    assert(all(abs.(λ[1]) < 1e-5))
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

"""
    sink(preds, weights=nothing)

Compute stationary distribution by sink method.

# Example

```jldoctest
julia> sink([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3-element Array{Float64,1}:
 1.0
 0.0
 0.0

julia> A = [0.5 0.2 0.1 0.3; 0.5 0.8 0.9 0.7]
2×4 Array{Float64,2}:
 0.5  0.2  0.1  0.3
 0.5  0.8  0.9  0.7

julia> B = [0.5 0.1 0.2 0.7; 0.5 0.9 0.8 0.3]
2×4 Array{Float64,2}:
 0.5  0.1  0.2  0.7
 0.5  0.9  0.8  0.3

julia> sink([(A, [1, 2]), (B, [2, 3])])
3×4 Array{Float64,2}:
 1.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0
 0.0  1.0  1.0  0.0

julia> sink([(A, [1, 2]), (B, [2, 3])], multiple_src=true)
3×4 Array{Float64,2}:
 0.333333  0.0  0.0  0.0
 0.333333  0.0  0.0  1.0
 0.333333  1.0  1.0  0.0
```
"""
function sink(preds, weights=nothing; multiple_src=false)
    ndims(preds[1][1]) == 1 &&
        return chase(build(preds, weights), 1)
    src = 1
    if multiple_src
        nclass = maximum(maximum.(getindex.(preds, 2)))
        src = 1:nclass
    end
    ans = []
    for i = 1:size(preds[1][1], 2)
        push!(ans, chase(build(map(x -> (x[1][:, i], x[2]), preds), weights), src))
    end
    hcat(ans...)
end

"""
    chase(Q::Array, src)

Chase to the highest points from src points in the graph with weights Q.
\$Q_{i,j}\$ is the weight going from point \$j\$ to \$i\$.
Returns an array indicates the number of each point being end points.

# Example

```jldoctest
julia> G = CTMCEnsemble.build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])]);

julia> CTMCEnsemble.chase(G, 1)
3-element Array{Float64,1}:
 1.0
 0.0
 0.0

julia> CTMCEnsemble.chase(G, [1, 2])
3-element Array{Float64,1}:
 0.5
 0.5
 0.0

julia> G = CTMCEnsemble.build([([0.4, 0.6], [1, 2]), ([0.4, 0.6], [2, 3])]);

julia> CTMCEnsemble.chase(G, [1, 2, 3])
3-element Array{Float64,1}:
 0.0
 0.0
 1.0
```
"""
function chase(Q::Array, src)
    nclass = size(Q, 1)
    assert(nclass == size(Q, 2) && maximum(src) <= nclass)
    p = zeros(nclass)
    d = Dict()
    inds = []
    for i in src
        at = i
        while true
            if haskey(d, at)
                at = d[at]
                break
            end
            push!(inds, at)
            v = Q[:, at] - Q[at, :]
            maximum(v) <= 1e-7 && break
            at = indmax(v)
        end
        for j in inds
            d[j] = at
        end
        p[at] += 1
    end
    normalize!(p, 1)
end

"""
    top1(data, label)

Top-1 accuracy.

# Example

```jldoctest
julia> p = [0.333333  0.0243902  0.0217391  0.230769
            0.333333  0.097561   0.195652   0.538462
            0.333333  0.878049   0.782609   0.230769];

julia> top1(p, [1, 3, 3, 2])
1.0
```
"""
top1(data, label) = mean(indmax.(view.([data], [:], 1:size(data, 2))) .== vec(label))

"""
    top5(data, label)

Top-5 accuracy.

# Example

```jldoctest
julia> p = [0.49  0.09  0.71  0.07  0.28
            0.73  0.48  0.01  0.96  0.51
            0.87  0.09  0.76  0.63  0.39
            0.37  0.65  0.89  0.31  0.42
            0.6   0.49  0.19  0.21  0.77
            0.56  0.32  0.27  1.0   0.92
            0.5   0.83  0.99  0.4   0.81
            0.34  0.03  0.83  0.07  0.62
            0.93  0.75  0.15  0.37  0.21
            0.25  0.19  0.83  0.69  0.64];

julia> top5(p, 1:5)
0.6
```
"""
top5(data, label) = mean(in.(vec(label), getindex.(sortperm.(view.([data], [:], 1:size(data, 2)), rev=true), [1:5])))

"""
    softmax!(data)

Inplace softmax col-wise.

# Example

```jldoctest
julia> p = [0.49  0.09  0.71  0.07  0.28
            0.73  0.48  0.01  0.96  0.51
            0.87  0.09  0.76  0.63  0.39
            0.37  0.65  0.89  0.31  0.42
            0.6   0.49  0.19  0.21  0.77
            0.56  0.32  0.27  1.0   0.92
            0.5   0.83  0.99  0.4   0.81
            0.34  0.03  0.83  0.07  0.62
            0.93  0.75  0.15  0.37  0.21
            0.25  0.19  0.83  0.69  0.64];

julia> softmax!(p)
10×5 Array{Float64,2}:
 0.0907496  0.0711453  0.109493   0.063526   0.0739451
 0.115365   0.10508    0.0543729  0.154694   0.0930673
 0.132702   0.0711453  0.115107   0.111213   0.0825433
 0.0804877  0.124552   0.131088   0.0807574  0.0850571
 0.101302   0.106136   0.0650961  0.0730723  0.120702
 0.0973297  0.0895435  0.0705178  0.161007   0.140235
 0.0916616  0.149116   0.144874   0.0883626  0.125628
 0.0781089  0.0670021  0.123454   0.063526   0.103889
 0.140908   0.137651   0.0625437  0.0857511  0.0689459
 0.0713862  0.0786277  0.123454   0.11809    0.105988
```
"""
function softmax!(data)
    m = maximum(data)
    data .= exp.(data .- m)
    normalize!.(view.([data], [:], 1:size(data, 2)), 1)
    data
end

end # module
