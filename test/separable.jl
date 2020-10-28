module TestSeparable
using Test
using Kernel
using Kernel: Separable, matmat2mat
using LinearAlgebra
using KroneckerProducts
k = Kernel.EQ()
@testset "SeparableKernel" begin
    d = 3
    B = randn(d, d)
    B = B'B
    G = Separable(k, B)
    @test G isa Separable

    @test G(0, 0) ≈ B

    n = 3
    x = randn(n)
    K = Kernel.gramian(G, x)
    @test issymmetric(K)
    @test isposdef(K)
    @test size(K) == (n, n)
    MK = matmat2mat(K)
    @test size(MK) == (d*n, d*n)

    KK = kronecker(K)
    @test size(KK) == size(MK)
    @test Matrix(KK) ≈ MK
end

end
