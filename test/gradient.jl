module TestGradient
using Test
using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Dot, ExponentialDot, NN, Lengthscale
using CovarianceFunctions: GradientKernel, ValueGradientKernel,
                           DerivativeKernel, ValueDerivativeKernel,
                           input_trait, BlockFactorization
using LinearAlgebra
const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

@testset "gradient kernels" begin
    # nd test
    n = 32
    d = 16
    # n = 2
    # d = 2
    X = randn(d, n) / sqrt(d)
    ε = 1e4eps()
    @testset "GradientKernel" begin
        kernels = [EQ(), RQ(1.), Dot()^3, ExponentialDot(), NN()]
        # kernels = [EQ(), Lengthscale(EQ(), 0.1)] # at this time, test doesn't work because fallback is incorrect
        a = randn(d*n)
        b = randn(d*n)
        for k in kernels
            G = GradientKernel(k)
            K = CovarianceFunctions.gramian(G, X)
            @test K isa BlockFactorization{<:Real}

            MK = Matrix(K)
            @test maximum(abs, MK - MK') < ε
            MK = Symmetric(MK)
            @test all(≥(-1e-12), eigvals(MK)) # positive semidefinite
            @test size(MK) == (d*n, d*n)

            # create anonymous wrapper around kernel to trigger generic fallback
            k2 = (x, y)->k(x, y)
            G2 = GradientKernel(k2)
            # G2(x[1], x[2]) # this takes centuries to precompile, maybe because of nested differentiation?
            # not really important because it is a fallback anyway
            K2 = CovarianceFunctions.gramian(G2, X)
            MK2 = Matrix(K2)
            @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel

            # testing matrix multiply
            Kab = deepcopy(b)
            α, β = randn(2) # trying out two-argument mul
            mul!(Kab, K, a, α, β) # saves some memory, Woodbury still allocates a lot
            MKab = @. α * $*(MK, a) + β * b
            @test Kab ≈ MKab
        end

        # testing matrix solve
        # println("div")
        for k in [EQ(), RQ(1.)] # TODO: better conditioned NN kernel with bias
            G = GradientKernel(k)
            K = CovarianceFunctions.gramian(G, X)
            a = randn(n*d)
            Ka = K*a
            as = K\Ka
            @test norm(K*as-Ka) / norm(Ka) < 1e-6
        end
    end

    @testset "ValueGradientKernel" begin
        kernels = [Dot(), Dot()^3] #[EQ(), RQ(1.)]#, Dot(), Dot()^3, NN()]
        a = randn((d+1)*n)
        b = randn((d+1)*n)
        for k in kernels
            G = ValueGradientKernel(k)
            K = CovarianceFunctions.gramian(G, X)
            @test K isa BlockFactorization{<:Real}

            MK = Matrix(K)
            @test maximum(abs, MK - MK') < ε
            MK = Symmetric(MK)
            if !all(≥(-1e-10), eigvals(MK))
                println("eigen-snafoo")
                display(eigvals(MK))
            end
            @test all(≥(-1e-10), eigvals(MK)) # positive semidefinite
            @test size(MK) == ((d+1)*n, (d+1)*n)

            # create anonymous wrapper around kernel to trigger generic fallback
            k2 = (x, y)->k(x, y)
            G2 = ValueGradientKernel(k2)
            # G2(x[1], x[2]) # this takes centuries to precompile, maybe because of nested differentiation?
            # not really important because it is a fallback anyway
            K2 = CovarianceFunctions.gramian(G2, X)
            MK2 = Matrix(K2)
            @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel

            # testing matrix multiply
            Kab = deepcopy(b)
            α, β = randn(2) # trying out two-argument mul
            mul!(Kab, K, a, α, β) # saves some memory, Woodbury still allocates a lot
            MKab = @. α * $*(MK, a) + β * b
            @test Kab ≈ MKab
        end

        # testing matrix solve
        # println("div")
        for k in [EQ(), RQ(1.)] # TODO: better conditioned NN kernel with bias
            G = ValueGradientKernel(k)
            K = CovarianceFunctions.gramian(G, X)
            a = randn(n*(d+1))
            Ka = K*a
            as = K\Ka
            @test norm(K*as-Ka) / norm(Ka) < 1e-6
        end
    end
    # matrix multiply
    # y = zeros(n*d)
    # mul!(y, K, x)

    # 1d special case
    @testset "ValueDerivativeKernel" begin
        k = CovarianceFunctions.EQ()
        G = ValueDerivativeKernel(k)
        n = 3
        x = randn(n)
        K = CovarianceFunctions.gramian(G, x)
        MK = Matrix(K)
        @test issymmetric(MK)
        @test all(≥(-1e-10), eigvals(MK)) # positive semidefinite
        @test size(K) == (2n, 2n)
        @test size(MK) == (2n, 2n)
    end
end

end
