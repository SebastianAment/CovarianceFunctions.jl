module TestMercer

using Test
using Kernel: iscov
using LinearAlgebra

@testset "promotion" begin
    using Kernel: Dot, Poly, NN
    k_arr = [Dot(), Poly(3), NN()]
    k_strings = ["Dot", "Poly", "NN"]
    for (k, str) in zip(k_arr, k_strings)
        @test k(1, 2.) ≈ k(1., 2.)
    end
end

# @testset "Mercer Kernels" begin
#     k = [Dot, Poly, NN]
#     @test
# end
#
# using Kernel: Product, Sum, Power, VerticalRescaling
# @testset "Kernel Algebra" begin
#     k = [Dot, Poly, NN]
#     @test
# end
#
# using RescaledMetric, Periodic, Sym, Conditional, SoR
# @testset "Kernel Modification" begin
#     k = [Dot, Poly, NN]
#     @test
# end

# @testset "Matrix Kernels" begin
#     k = [Dot, Poly, NN]
#     @test
# end

#
# # projection onto symmetric matrices: A -> (A' + A)/2
# # testing if the rescaled kernel is numerically stable
# function test(k::K, b::T = T(1), n::Int = 1024) where {T, K<:Rescaled{T}}
#     ε = eps(T) # numerical precision of type
#     x = 2b * rand(n) .- b # randomly sample in range
#     ax = k.a.(x)
#     cond = maximum(ax) / minimum(ax)
#     if cond > sqrt(1/ε)
#         println("Warning: Currently specified Rescaled kernel leads to ill-conditioned GP inference. To avoid this, we added 1e-6 to your rescaling function.")
#         f(x) = k.a(x) + T(1e-6)
#         return K(k.k, f)
#     end
#     return k
# end

end # module TestMercer
