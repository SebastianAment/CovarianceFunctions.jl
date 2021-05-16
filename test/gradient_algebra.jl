module TestGradientAlgebra
using Test
using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Dot, ExponentialDot, NN
using CovarianceFunctions: GradientKernel, ValueGradientKernel,
                           DerivativeKernel, ValueDerivativeKernel,
                           input_trait, BlockFactorization
using WoodburyIdentity
using LinearAlgebra
using LazyLinearAlgebra
const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

@testset "input trait property" begin
    # dot product
    k = CovarianceFunctions.Dot()^5
    h = CovarianceFunctions.Dot()^2 + 1 # quadratic
    @test input_trait(k+h) == DotProductInput()
    @test input_trait(k*h) == DotProductInput()

    # isotropic
    k = CovarianceFunctions.EQ()
    h = CovarianceFunctions.RQ(1.0) # quadratic
    @test input_trait(k+h) == IsotropicInput()
    @test input_trait(k*h) == IsotropicInput()

    # heterogeneous
    k = CovarianceFunctions.EQ()
    h = CovarianceFunctions.Dot()^2 # quadratic
    @test input_trait(k+h) == GenericInput()
    @test input_trait(k*h) == GenericInput()
end

@testset "gradient algebra" begin
    # test data
    d, n = 3, 7
    X = randn(d, n)

    # sum of homogeneous kernel types
    k = CovarianceFunctions.Dot()^5
    h = CovarianceFunctions.Dot()^2 + 1 # quadratic
    gk, gh = GradientKernel(k), GradientKernel(h)
    G = GradientKernel(k + h)
    K, Kk, Kh = gramian(G, X), gramian(gk, X), gramian(gh, X)
    MK, Mkk, Mkh = Matrix(K), Matrix(Kk), Matrix(Kh)
    @test K isa BlockFactorization
    @test K.A[1, 1] isa Woodbury # this means it correctly consolidated the sum kernel into one dot product kernel
    @test MK ≈ Mkk + Mkh

    # sum of heterogeneous kernel types
    k = CovarianceFunctions.EQ() # EQ
    h = CovarianceFunctions.Dot()^2 + 1 # quadratic
    gk, gh = GradientKernel(k), GradientKernel(h)
    G = GradientKernel(k + h)
    K, Kk, Kh = gramian(G, X), gramian(gk, X), gramian(gh, X)

    MK, Mkk, Mkh = Matrix(K), Matrix(Kk), Matrix(Kh)
    @test K isa BlockFactorization
    @test K.A[1, 1] isa LazyMatrixSum
    @test MK ≈ Mkk + Mkh
    a = randn(d*n)
    @test K*a ≈ MK*a
    @test K*a ≈ Mkk*a + Mkh*a
    @test K*a ≈ Kk*a + Kh*a
end

end
