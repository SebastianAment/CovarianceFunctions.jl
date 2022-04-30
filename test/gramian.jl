module TestGramian
using CovarianceFunctions
using Test
using LinearAlgebra
using ToeplitzMatrices
using CovarianceFunctions
using CovarianceFunctions: PeriodicInput, LazyMatrixSum
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
    k = CovarianceFunctions.EQ()
    G = gramian(k, x)
    # type promotion
    @test k(x[1], x[2]) isa typeof(G[1, 2]) # test promotion in inner constructor
    @test k(x[1], x[2]) isa eltype(G) # test promotion in inner constructor
    x = tuple.(x, x)
    G = gramian(k, x)
    @test k(x[1], x[2]) isa typeof(G[1,1])  # with tuples (or arrays)
    @test k(x[1], x[2]) isa eltype(G)

    # adding diagonal to Gramian is done lazily
    D = 1e-6I(n)
    @test D + G isa LazyMatrixSum
    @test G + D isa LazyMatrixSum

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

end

@testset "Gramian factorization" begin
    rtol = 1e-6
    k = CovarianceFunctions.EQ()
    n = 64
    x = randn(n)
    G = gramian(k, x)
    F = factorize(G)
    @test F isa CholeskyPivoted
    @test rank(F) < n ÷ 2 # this should be very low rank
    @test isapprox(Matrix(F), G, rtol = rtol)

    # pivoted cholesky
    G = gramian(k, x)
    F = cholesky(G, Val(true), check = false, tol = 1e-6)
    @test F isa CholeskyPivoted
    @test isapprox(Matrix(F), G, rtol = rtol)

    # regular cholesky
    k = CovarianceFunctions.Exponential() # does not yield low rank matrix
    G = gramian(k, x)
    F = cholesky(G)
    @test F isa Cholesky
    @test isapprox(Matrix(F), G, rtol = rtol)

    # testing gramian of matrix-valued anonymous kernel
    d = 3
    A = randn(d, d)
    k = (x, y)->A
    G = gramian(k, x)
    GA = Matrix(G)
    a = randn(d*n)
    @test size(G) == (d*n, d*n)
    @test G isa BlockFactorization
    @test G*a ≈ GA*a

    # adding diagonal by default maintains laziness
    @test I(d*n) + G isa LazyMatrixSum
    @test G + I(d*n) isa LazyMatrixSum

    # same thing for block diagonal matrices
    D = BlockFactorization(Diagonal([randn(d, d) for _ in 1:n]))
    @test D + G isa LazyMatrixSum
    @test G + D isa LazyMatrixSum
end

@testset "toeplitz structure" begin
    n = 32
    x = range(-1, 1, n)
    k = CovarianceFunctions.EQ()
    G = gramian(k, x)
    @test G isa SymmetricToeplitz
    @test size(G) == (n, n)
    @test G*x ≈ Matrix(G)*x

    @test Matrix(G) ≈ Matrix(Gramian(k, x))
    G = gramian(k, x, PeriodicInput()) # periodic boundary conditions
    @test G isa Circulant
    @test size(G) == (n, n)
    @test G*x ≈ Matrix(G)*x
    # @test Matrix(G) ≈ Matrix(Gramian(k, x))
end # test solves and multiplications

end # TestGramian
