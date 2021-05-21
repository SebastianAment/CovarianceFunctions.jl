module TestGramian
using CovarianceFunctions
using Test
using LinearAlgebra
using ToeplitzMatrices
using CovarianceFunctions: gramian, Gramian
using LazyLinearAlgebra: BlockFactorization

@testset "Gramian properties" begin
    n = 8
    x = randn(Float64, n)
    k = CovarianceFunctions.EQ()
    G = gramian(k, x)

    @test issymmetric(G)
    M = Matrix(G)
    @test M ≈ k.(x, x')
    @test issymmetric(G) && issymmetric(M)
    @test isposdef(G) && isposdef(M) # if M is very low-rank, this can fail -> TODO: ispsd()

    G = gramian(k, x, x .+ randn(size(x)))
    M = Matrix(G)
    @test !issymmetric(G) && !issymmetric(M)
    @test !isposdef(G) && !isposdef(M)

    G = gramian(k, x, copy(x))
    @test issymmetric(G) # testing that value check also works
    @test isposdef(G)

    # testing standard Gramian
    G = Gramian(x, x)
    @test G ≈ dot.(x, x')

    x = randn(Float64, n)
    k = CovarianceFunctions.EQ{Float32}()
    G = gramian(k, x)
    # type promotion
    @test typeof(k(x[1], x[1])) <: typeof(G[1,1]) # test promotion in inner constructor
    @test typeof(k(x[1], x[1])) <: eltype(G) # test promotion in inner constructor
    x = tuple.(x, x)
    G = gramian(k, x)
    @test typeof(k(x[1], x[1])) <: typeof(G[1,1])  # with tuples (or arrays)
    @test typeof(k(x[1], x[1])) <: eltype(G)

    # TODO: MultiKernel test
    k = (x, y) -> randn(3, 2)
    G = gramian(k, x)
    @test G isa BlockFactorization # testing that matrix-valued kernel returns block factorization
    @test size(G) == (3n, 2n)
end

@testset "Gramian factorization" begin
    k = CovarianceFunctions.EQ()
    n = 64
    x = randn(n)
    G = gramian(k, x)
    F = factorize(G)
    @test F isa CholeskyPivoted
    @test issuccess(F)
    @test rank(F) < n ÷ 2 # this should be very low rank
    @test isapprox(Matrix(F), G, atol = 1e-8)

    F = cholesky(G)
    @test F isa CholeskyPivoted
    @test isapprox(Matrix(F), G, atol = 1e-8)

    # regular cholesky
    k = CovarianceFunctions.Exponential() # does not yield low rank matrix
    G = gramian(k, x)
    F = cholesky(G, Val(false))
    @test F isa Cholesky
    @test isapprox(Matrix(F), G, atol = 1e-8)
end

@testset "toeplitz structure" begin
    n = 16
    x = -1:.1:1
    k = CovarianceFunctions.EQ()
    m = gramian(k, x)
    # @test typeof(m) <: SymmetricToeplitz
    m = gramian(k, x, Val(true)) # periodic boundary conditions
    @test m isa Circulant
end

end # TestGramian
