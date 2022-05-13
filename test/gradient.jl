module TestGradient
using Test
using LinearAlgebra
using BlockFactorizations

using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Dot, ExponentialDot, NN, Matern, MaternP,
        Lengthscale, input_trait, GradientKernel, ValueGradientKernel, GradientKernelElement,
        DerivativeKernel, ValueDerivativeKernel, DerivativeKernelElement, Cosine,
        Woodbury, LazyMatrixProduct, ConstantKernel

const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

@testset "gradient kernels" begin
    # nd test
    n = 2
    d = 5
    X = randn(d, n) / sqrt(d)
    ε = 1e4eps()
    @testset "GradientKernel" begin
        kernels = [MaternP(3), Dot()^3, Cosine(randn(d)), NN()]
        # kernels = [EQ(), RQ(1.), MaternP(2), Matern(2.7), Dot()^3, ExponentialDot(), NN(), Cosine(randn(d))]
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

            # this takes centuries to precompile, maybe because of nested differentiation?
            # not really important because it is a fallback anyway
            K2 = CovarianceFunctions.gramian(G2, X) # NOTE: this is the bottleneck in test suite
            MK2 = Matrix(K2)
            @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel

            # testing matrix multiply
            Kab = deepcopy(b)
            α, β = randn(2) # trying out two-argument mul!
            mul!(Kab, K, a, α, β) # saves some memory, Woodbury still allocates a lot
            MKab = @. α * $*(MK, a) + β * b
            @test Kab ≈ MKab
        end

        # testing matrix solve
        for k in kernels # TODO: better conditioned NN kernel with bias
            G = GradientKernel(k)
            K = CovarianceFunctions.gramian(G, X)
            a = randn(n*d)
            Ka = K * a
            as = K \ Ka
            @test norm(K*as-Ka) / norm(Ka) < 1e-6
        end

        # testing non-lazy data-sparse representation
        x, y = X[:, 1], X[:, 2]
        a = randn(length(x))
        G = GradientKernelElement(EQ(), x, y)
        W = Woodbury(G)
        @test W*a ≈ G*a

        G = GradientKernelElement(Dot()^3, x, y)
        W = Woodbury(G)
        @test W*a ≈ G*a

        G = GradientKernelElement(Cosine(randn(d)), x, y)
        W = LazyMatrixProduct(G)
        @test W*a ≈ G*a

        # testing constant kernel
        c = ConstantKernel(1)
        g = GradientKernel(c)
        @test g(x, y) ≈ zeros(d, d)
    end

    # TODO:
    @testset "ValueGradientKernel" begin
        kernels = [MaternP(3), Dot()^3, Cosine(randn(d))]
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
            K2 = CovarianceFunctions.gramian(G2, X) # NOTE: compilation here is super slow (not important for real applicaiton since it is fallback)
            MK2 = Matrix(K2)
            @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel

            # instead, compare to already tested GradientKernel
            # K12 = K.A[1, 2]
            # @test K12 isa DerivativeKernelElement
            # @test K12.value_value ≈ k(X[:, 1], X[:, 2])
            # @test Matrix(K12.gradient_gradient) ≈ Matrix(GradientKernel(k)(X[:, 1], X[:, 2]))

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
