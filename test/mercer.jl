module TestMercer

using Test
using CovarianceFunctions: iscov
using LinearAlgebra

@testset "promotion" begin
    using CovarianceFunctions: Dot, Poly, NN, ExponentialDot
    k_arr = [Dot(), Poly(3), NN()]
    k_strings = ["Dot", "Poly", "NN"]
    for (k, str) in zip(k_arr, k_strings)
        @test k(1, 2.) ≈ k(1., 2.)
    end
    d = 3
    x, y = randn(d), randn(d)
    @test (Dot()^3)(x, y) ≈ Poly(3)(x, y)
    @test ExponentialDot()(x, y) ≈ exp(dot(x, y))
end

@testset "mercer" begin
    using CovarianceFunctions: FiniteBasis, gramian, Gramian
    using LinearAlgebraExtensions: LowRank

    basis = [sin, cos, identity]
    k = FiniteBasis(basis)
    @test k isa FiniteBasis
    x = randn(16)
    @test gramian(k, x) isa LowRank
    U = [sin.(x) cos.(x) x]
    @test Matrix(gramian(k, x)) ≈ U*U'

    x = randn(2)
    @test gramian(k, x) isa Gramian # if # of functions is larger than data
    U = [sin.(x) cos.(x) x]
    @test Matrix(gramian(k, x)) ≈ U*U'

    # FiniteBasis([]) # throws
end

# @testset "Mercer Kernels" begin
#     k = [Dot, Poly, NN]
#     @test
# end
#
# using CovarianceFunctions: Product, Sum, Power, VerticalRescaling
# @testset "CovarianceFunctions Algebra" begin
#     k = [Dot, Poly, NN]
#     @test
# end
#
# using RescaledMetric, Periodic, Sym, Conditional, SoR
# @testset "CovarianceFunctions Modification" begin
#     k = [Dot, Poly, NN]
#     @test
# end

# @testset "Matrix Kernels" begin
#     k = [Dot, Poly, NN]
#     @test
# end

#
# # projection onto symmetric matrices: A -> (A' + A)/2
# # testing if the rescaled CovarianceFunctions is numerically stable
# function test(k::K, b::T = T(1), n::Int = 1024) where {T, K<:Rescaled{T}}
#     ε = eps(T) # numerical precision of type
#     x = 2b * rand(n) .- b # randomly sample in range
#     ax = k.a.(x)
#     cond = maximum(ax) / minimum(ax)
#     if cond > sqrt(1/ε)
#         println("Warning: Currently specified Rescaled CovarianceFunctions leads to ill-conditioned GP inference. To avoid this, we added 1e-6 to your rescaling function.")
#         f(x) = k.a(x) + T(1e-6)
#         return K(k.k, f)
#     end
#     return k
# end

end # module TestMercer
