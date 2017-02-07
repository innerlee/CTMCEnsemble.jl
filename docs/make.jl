using Documenter, CTMCEnsemble

makedocs(
    modules=[CTMCEnsemble]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/innerlee/CTMCEnsemble.jl.git",
    julia  = "release",
    osname = "osx"
)
