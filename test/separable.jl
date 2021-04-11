module TestSeparable
using Test
using CovarianceFunctions
using CovarianceFunctions: Separable
using LinearAlgebra
using KroneckerProducts
k = CovarianceFunctions.EQ()
@testset "SeparableKernel" begin
    d = 3
    B = randn(d, d)
    B = B'B
    G = Separable(k, B)
    @test G isa Separable

    @test G(0, 0) ≈ B

    n = 3
    x = randn(n)
    K = CovarianceFunctions.gramian(G, x)
    @test size(K) == (d*n, d*n)
    MK = Matrix(K)
    @test issymmetric(MK)
    @test isposdef(MK)
    @test size(MK) == (d*n, d*n)

    KK = kronecker(K)
    @test size(KK) == size(MK)
    @test Matrix(KK) ≈ MK
end

end
