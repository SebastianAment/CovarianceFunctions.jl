module TestGramian
using CovarianceFunctions
using Test
using LinearAlgebra
using ToeplitzMatrices
using CovarianceFunctions
using CovarianceFunctions: PeriodicInput
using BlockFactorizations

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

    # rectangular multiply
    x = randn(n)
    y = randn(2n)
    G = gramian(k, x, y)
    @test size(G) == (n, 2n)
    a = randn(2n)
    b = G*a
    @test length(b) == n
    @test b ≈ Matrix(G) * a

    y = randn(2n)
    G = gramian(k, x, y)
    @test size(G) == (n, 2n)
    p = 3
    A = randn(2n, p)
    B = G*A
    @test size(B) == (n, p)
    @test B ≈ Matrix(G) * A


    # TODO: MultiKernel test
    k = (x, y) -> randn(3, 2)
    G = gramian(k, x)
    @test G isa BlockFactorization # testing that matrix-valued kernel returns block factorization
    @test size(G) == (3n, 2n)
end

@testset "Gramian factorization" begin
    atol = 1e-8
    k = CovarianceFunctions.EQ()
    n = 64
    x = randn(n)
    G = gramian(k, x)
    F = factorize(G)
    @test F isa CholeskyPivoted
    @test rank(F) < n ÷ 2 # this should be very low rank
    @test isapprox(Matrix(F), G, atol = atol)

    F = cholesky(G)
    @test F isa CholeskyPivoted
    @test isapprox(Matrix(F), G, atol = atol)

    # regular cholesky
    k = CovarianceFunctions.Exponential() # does not yield low rank matrix
    G = gramian(k, x)
    F = cholesky(G, Val(false))
    @test F isa Cholesky
    @test isapprox(Matrix(F), G, atol = atol)

    # testing gramian of matrix-valued anonymous kernel
    A = randn(3, 2)
    k = (x, y)->A
    G = gramian(k, x)
    GA = Matrix(G)
    a = randn(2*n)
    @test G isa BlockFactorization
    @test G*a ≈ GA*a
end

@testset "toeplitz structure" begin
    n = 32
    x = range(-1, 1, n)
    k = CovarianceFunctions.EQ()
    G = gramian(k, x)
    @test G isa SymmetricToeplitz
    @test size(G) == (n, n)
    @test Matrix(G) ≈ Matrix(Gramian(k, x))
    G = gramian(k, x, PeriodicInput()) # periodic boundary conditions
    @test G isa Circulant
    @test size(G) == (n, n)
    # @test Matrix(G) ≈ Matrix(Gramian(k, x))
end # TODO: test solves and multiplications

end # TestGramian
