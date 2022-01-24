module TestSparse
using LinearAlgebra
using CovarianceFunctions
using SparseArrays
using Test

@testset "sparse" begin
    n = 128
    d = 32
    X = randn(d, n)
    k = CovarianceFunctions.EQ()
    δ = 1e-6
    G = gramian(k, X)
    S = sparse(G, δ)
    @test isapprox(G, S, atol = δ, rtol = δ)
end

end # module
