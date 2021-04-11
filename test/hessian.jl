module TestHessian
using Test
using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Dot, NN
using CovarianceFunctions: HessianKernel, ValueGradientHessianKernel,
                           input_trait, BlockFactorization,
                           HessianKernelElement, ValueGradientHessianKernelElement
using LinearAlgebra
const AbstractMatOrFac = Union{AbstractMatrix, Factorization}

@testset "Hessian kernels" begin
    # nd test
    n = 2
    d = 3
    X = randn(d, n)
    ε = 2eps()
    # @testset "HessianKernelElement" begin
    #     kernels = [EQ(), RQ(1.), Dot()^2, Dot()^4]# Dot()^5] #, NN()]
    #     for k in kernels
    #         # K = HessianKernelElement(k, X[:, 1], X[:, 2])
    #         K = HessianKernel(k)(X[:, 1], X[:, 2])
    #         @test K isa HessianKernelElement
    #         H = HessianKernel((x, y) -> k(x, y))(X[:, 1], X[:, 2])
    #         @test H isa Matrix{<:Real}
    #         @test size(K) == (d^2, d^2)
    #         @test Matrix(K) ≈ H
    #         a, b = randn(d^2), randn(d^2)
    #         α, β = randn(2)
    #         c = α * H * a + β * b
    #         mul!(b, K, a, α, β)
    #         @test b ≈ c
    #     end
    # end
    #
    # @testset "HessianKernel" begin
    #     aa = randn(n*d^2)
    #     bb = randn(n*d^2)
    #     kernels = [EQ(), RQ(1.), Dot()^3, Dot()^4] #, NN()]
    #     for k in kernels
    #         println(k)
    #         G = HessianKernel(k)
    #         K = CovarianceFunctions.gramian(G, X)
    #         @test K isa BlockFactorization{<:Real}
    #
    #         MK = Matrix(K)
    #         @test maximum(abs, MK - MK') < ε
    #         MK = Symmetric(MK)
    #         @test all(≥(-10maximum(MK)*eps()), eigvals(MK)) # positive semidefinite
    #         @test size(MK) == (d^2*n, d^2*n)
    #
    #         # create anonymous wrapper around kernel to trigger generic fallback
    #         k2 = (x, y)->k(x, y)
    #         G2 = HessianKernel(k2)
    #         # G2(x[1], x[2]) # this takes centuries to precompile, maybe because of nested differentiation?
    #         # not really important because it is a fallback anyway
    #         K2 = CovarianceFunctions.gramian(G2, X)
    #         MK2 = Matrix(K2)
    #         @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel
    #
    #         # testing matrix multiply
    #         println("mul")
    #         Kab = deepcopy(bb)
    #         α, β = randn(2) # trying out two-argument mul
    #         @time mul!(Kab, K, aa, α, β) # saves some memory, Woodbury still allocates a lot
    #         @time MKab = @. α * $*(MK, aa) + β * bb
    #         # Kab = K*aa
    #         # MKab = Matrix(K)*aa
    #         @test Kab ≈ MKab
    #     end
    # end

    dd = d^2 + d + 1
    @testset "ValueGradientHessianKernelElement" begin
        kernels = [EQ()] #, RQ(1.), Dot()^2, Dot()^4]# Dot()^5] #, NN()]
        for k in kernels
            K = ValueGradientHessianKernel(k)(X[:, 1], X[:, 2])
            @test K isa ValueGradientHessianKernelElement
            H = ValueGradientHessianKernel((x, y) -> k(x, y))(X[:, 1], X[:, 2])
            @test H isa Matrix{<:Real}
            @test size(K) == (dd, dd)
            @test Matrix(K) ≈ H

            # @test b ≈ c

            # block indices
            vi = 1 # value index
            gi = (1+1):(d+1) # gradient index
            hi = (d+1+1):(1+d+d^2) # hessian index

            a = randn(dd)
            # a[vi] = 0 # used to debug individual components
            # @. a[gi] = 0
            # @. a[hi] = 0
            b = randn(dd)
            α, β = randn(2)

            c = α * H * a + β * b
            mul!(b, K, a, α, β)
            @test b[vi] ≈ c[vi]
            @test b[gi] ≈ c[gi]
            @test b[hi] ≈ c[hi]
        end
    end

    # @testset "ValueGradientHessianKernel" begin
    #     kernels = [EQ()] #, RQ(1.), Dot(), Dot()^3, NN()]
    #     ε = 1e4eps()
    #     aa = randn(n*dd)
    #     bb = randn(n*dd)
    #     for k in kernels
    #         println(k)
    #         G = ValueGradientHessianKernel(k)
    #         display(G)
    #         K = CovarianceFunctions.gramian(G, X)
    #         @test K isa BlockFactorization{<:Real}
    #
    #         MK = Matrix(K)
    #         @test maximum(abs, MK - MK') < ε
    #         MK = Symmetric(MK)
    #         @test all(≥(-1e4eps()), eigvals(MK)) # positive semidefinite
    #         @test size(MK) == (dd*n, dd*n)
    #         # # create anonymous wrapper around kernel to trigger generic fallback
    #         # k2 = (x, y)->k(x, y)
    #         # G2 = ValueGradientKernel(k2)
    #         # # G2(x[1], x[2]) # this takes centuries to precompile, maybe because of nested differentiation?
    #         # # not really important because it is a fallback anyway
    #         # K2 = CovarianceFunctions.gramian(G2, X)
    #         # MK2 = Matrix(K2)
    #         # @test MK ≈ MK2 # compare generic and specialized evaluation of gradient kernel
    #         #
    #         # testing matrix multiply
    #         println("mul")
    #         Kab = deepcopy(bb)
    #         α, β = randn(2) # trying out two-argument mul
    #         # @time mul!(Kab, K, aa, α, β) # saves some memory, Woodbury still allocates a lot
    #         # @time MKab = @. α * $*(MK, aa) + β * bb
    #         Kab = K*aa
    #         MKab = Matrix(K)*aa
    #         # block indices
    #         vi = 1 # value index
    #         gi = (1+1):(d+1) # gradient index
    #         hi = (d+1+1):(1+d+d^2) # hessian index
    #         @test Kab[vi] ≈ MKab[vi]
    #         @test Kab[gi] ≈ MKab[gi]
    #         @test Kab[hi] ≈ MKab[hi]
    #         display(Kab[hi])
    #         display(MKab[hi])
    #         display(reshape(Kab[hi], d, d))
    #         display(reshape(MKab[hi], d, d))
    #     end
    # end
end

end # module
