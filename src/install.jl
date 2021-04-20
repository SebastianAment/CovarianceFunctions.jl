using Pkg

ssh = true
if ssh
    git = "git@github.com:SebastianAment/"
else
    git = "https://github.com/SebastianAment/"
end

add(x) = Pkg.add(Pkg.PackageSpec(url = git * x * ".git"))

add("LazyInverse.jl")
add("LazyLinearAlgebra.jl")
add("LinearAlgebraExtensions.jl")
add("KroneckerProducts.jl")
add("WoodburyIdentity.jl")
