module TestAlgebra

using Test
using LinearAlgebra
using CovarianceFunctions: AbstractKernel, IsotropicKernel, ismercer, isstationary, isisotropic

using CovarianceFunctions
using CovarianceFunctions: Constant, EQ, RQ, Exp, γExp, Delta, Cosine, MaternP, Matern #, SM
using CovarianceFunctions: separable, gramian, LazyGrid
using KroneckerProducts: KroneckerProduct

# TODO:
# test SymmetricKernel
# test higher input dimensions
# test type stability
@testset "stationary kernels" begin

    k1 = EQ()
    k2 = Matern(5*rand())
    k3 = CovarianceFunctions.Dot()

    @test typeof(k1 * k3) <: AbstractKernel
    @test ismercer(k1 * k3)
    # evaluations
    x = randn()
    y = randn()
    # sum
    @test k1(x, y) + k2(x, y) ≈ (k1+k2)(x, y)
    @test k1(x, y) + k2(x, y) ≈ (k2+k1)(x, y)
    @test k1(x, y) + k3(x, y) ≈ (k1+k3)(x, y)
    @test k1(x, y) + k3(x, y) ≈ (k3+k1)(x, y)

    # product
    @test k1(x, y) * k2(x, y) ≈ (k1*k2)(x, y)
    @test k1(x, y) * k2(x, y) ≈ (k2*k1)(x, y)
    @test k1(x, y) * k3(x, y) ≈ (k1*k3)(x, y)
    @test k1(x, y) * k3(x, y) ≈ (k3*k1)(x, y)

    # scalar test
    a = exp(randn())
    @test a*k1(x, y) ≈ (a*k1)(x, y)
    @test a*k1(x, y) ≈ (k1*a)(x, y)
    @test a + k1(x, y) ≈ (a+k1)(x, y)
    @test a + k1(x, y) ≈ (k1+a)(x, y)

    # power
    for p = 1:4
        @test k1(x, y)^p ≈ (k1^p)(x, y)
        @test k2(x, y)^p ≈ (k2^p)(x, y)
        @test k3(x, y)^p ≈ (k3^p)(x, y)
    end

    # TODO: Test invalid inputs for Power kernels
    # k_strings = ["Exponentiated Quadratic", "Exponential", "δ",
    #             "Constant", "Rational Quadratic",
    #             "γ-Exponential", "Cosine",
    #             "Matern"]#, "Spectral Mixture"]

    # T = Float64
    # k_arr = [EQ(), Exp(), Delta(),
    #         Constant(r), RQ(r),
    #         γExp(r), Cosine(r),
    #         Matern(r)]#, SM]
    # n = 16
    # d = 1
    # x = randn(T, n)
    # Σ = zeros(T, (n, n))
end

@testset "separable kernels" begin
    n = 4
    d = 3
    a, b = randn(d), randn(d)
    x = randn(n)
    gx = LazyGrid(x, d)
    y = randn(2n) # this applies the second kernel
    gy = LazyGrid(y, d)

    k = CovarianceFunctions.EQ()
    p3 = separable(*, (k for _ in 1:d)...)
    @test p3(a, b) ≈ separable(^, k, d)(a, b)
    @test gramian(p3, gx, gy) isa KroneckerProduct
    @test size(gramian(p3, gx, gy)) == (n^d, (2n)^d)

    @test p3(a, b) ≈ k(a, b) # since EQ is separable in dimension

    G = gramian(p3, gx, gy)
    @test G isa KroneckerProduct
    @test Matrix(G) ≈ p3.(gx, permutedims(gy))

    # separable sum
    s3 = separable(+, k, k, k)
    Gs3 = gramian(s3, gx, gy)
    @test Gs3 isa Gramian
    @test Matrix(Gs3) ≈ s3.(gx, permutedims(gy))

end

end # TestAlgebra

# moving to trait-based system ...
# @test typeof(k1 + k2) <: IsotropicKernel
# @test typeof(k1 * k2) <: IsotropicKernel
# @test typeof(k2 * k1) <: IsotropicKernel
# for p = 2:4
#     @test typeof(k1^p) <: IsotropicKernel
#     @test typeof(k2^p) <: IsotropicKernel
#     @test typeof(float(p)*k1) <: IsotropicKernel
# end
