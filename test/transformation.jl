module TestTransformation
using Test
using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Dot, NN, ScaledInputKernel
using CovarianceFunctions: GradientKernel, ValueGradientKernel,
                           DerivativeKernel, ValueDerivativeKernel, matmat2mat,
                           input_trait

using LinearAlgebra

@testset "input transformation" begin
    # nd test
    d = 3
    x, y = randn(d), randn(d)
    ε = 2eps()
    U = randn(d, d)
    kernels = [EQ(), RQ(1.), Dot(), Dot()^3, NN()]
    # testing evaluation
    for k in kernels
        s = ScaledInputKernel(k, U)
        @test s(x, y) ≈ k(U*x, U*y)
    end

    # testing gramian
    n = 5
    X = randn(d, n)
    for k in kernels
        s = ScaledInputKernel(k, U)
        G = gramian(s, X)
        @test typeof(G.k) == typeof(k)
        @test all((z) -> z[1] ≈ U*z[2], zip(G.x, eachcol(X)))
    end
    U = Diagonal(exp.(randn(d))) # diagonal does not need this special behavior
    for k in kernels
        s = ScaledInputKernel(k, U)
        G = gramian(s, X)
        @test typeof(G.k) == typeof(s)
        @test all((z) -> z[1] ≈ z[2], zip(G.x, eachcol(X)))
    end

    k = RQ(1.)
    s = ScaledInputKernel(k, U)
    g = GradientKernel(s)
    G = gramian(g, X)

    v = [randn(d) for _ in 1:n]

    # display(G)
    # Matrix(G)
end

end
