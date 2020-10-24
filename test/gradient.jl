module TestGradient
using Test
using Kernel
using Kernel: GradientKernel, matmat2mat
using LinearAlgebra

k = Kernel.EQ()
G = GradientKernel(k)

@testset "GradientKernel" begin
    # 1d test
    n = 3
    x = randn(n)
    K = Kernel.gramian(G, x)

    @test issymmetric(K)
    @test isposdef(K)
    @test size(K) == (n, n)
    MK = matmat2mat(K)
    @test size(MK) == (2n, 2n)

    K1 = copy(K)

    # nd test
    n = 2
    d = 1
    x = [randn(d) for _ in 1:n]
    K = Kernel.gramian(G, x)
    @test K isa AbstractMatrix{<:Matrix}

    MK = matmat2mat(K)
    ε = 2eps(eltype(eltype(MK)))
    @test maximum(MK - MK') < ε
    @test isposdef(MK)
    @test size(MK) == ((d+1)*n, (d+1)*n)

end

# l = .5
# k = Kernel.Lengthscale(Kernel.EQ(), l)
# dK = derivative_kernel(k)
#
# x = range(-1, 1, length = 2)
# K = derivative_kernel_matrix(k, x)
# @test all(>(0), eigvals(K))
#
# σ = 1e-2
# f(x) = x*sin(2π*x) + x^2
# g(x) = f(x) + σ*randn()
# df(x) = derivative(f, x)
# dg(x) = df(x) + σ*randn()
# x = randn(8)


end
