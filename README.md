# CTMCEnsemble

[![Build Status](https://travis-ci.com/innerlee/CTMCEnsemble.jl.svg?token=QaB6ijkBZpoUGF1MyQpy&branch=master)](https://travis-ci.com/innerlee/CTMCEnsemble.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://innerlee.github.io/CTMCEnsemble.jl/latest)

Implementation of the paper [Integrating Specialized Classifiers Based on Continuous Time Markov Chain](https://www.ijcai.org/proceedings/2017/312) by Zhizhong Li and Dahua Lin.

## Example

```julia
julia> G = build([([0.5, 0.5], [1, 2]), ([0.5, 0.5], [2, 3])])
3×3 Array{Float64,2}:
 -0.5   0.5   0.0
  0.5  -1.0   0.5
  0.0   0.5  -0.5

julia> stationdist(G)
3-element Array{Float64,1}:
 0.333333
 0.333333
 0.333333
```
