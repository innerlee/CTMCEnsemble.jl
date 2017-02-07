module CTMCEnsemble

"""

Build the generator matrix.
"""
function build(preds, nclass=0)
    nclass == 0 &&
        (nclass = maximum(getindex.(pres, 2)))

    for (pred, label) in preds

    end

end


end # module
