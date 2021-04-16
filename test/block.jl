module TestBlock
using CovarianceFunctions: BlockFactorization
using LinearAlgebra
using Test

@testset "block" begin
    di = 2
    dj = 3
    n = 5
    m = 7
    # strided
    A = [randn(di, dj) for i in 1:n, j in 1:m]
    F = BlockFactorization(A)
    M = Matrix(F)
    @test size(F) == (n*di, m*dj)
    for ni in 2:2, mi in 1:1
        for i in 1:di, j in 1:dj
            @test F[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
            @test M[di*(ni-1)+i, dj*(mi-1)+j] == A[ni, mi][i, j]
        end
    end

    # matrix multiply
    x = randn(size(M, 2))
    @test M*x ≈ F*x
    k = 3
    X = randn(size(M, 2), k)
    @test M*X ≈ F*X

    # general
    n, m = 4, 5
    nindices = [1, 2, n+1]
    mindices = [1, 4, m+1]
    A = fill(randn(0, 0), 2, 2)
    A[1, 1] = randn(1, 3)
    A[1, 2] = randn(1, 2)
    A[2, 1] = randn(3, 3)
    A[2, 2] = randn(3, 2)
    F = BlockFactorization(A, nindices, mindices)
    M = Matrix(F)
    @test size(F) == (n, m)
    for i in 1:4, j in 1:5
        @test F[i, j] == M[i, j]
    end

    # matrix multiply
    x = randn(size(M, 2))
    @test M*x ≈ F*x
    k = 3
    X = randn(size(M, 2), k)
    @test M*X ≈ F*X
end

end
