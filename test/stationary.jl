module TestStationary

using Test
using LinearAlgebra
using LinearAlgebraExtensions: LowRank

using CovarianceFunctions
using CovarianceFunctions: MercerKernel, StationaryKernel, isstationary, isisotropic,
                    Constant, EQ, RQ, Exp, γExp, Delta, Cosine, MaternP, Matern#, SM
using CovarianceFunctions: iscov, enorm, Normed
using ForwardDiff: derivative

# isotropic kernels
const iso_k_strings = ["Exponentiated Quadratic", "Exponential", "δ",
            "Constant", "Rational Quadratic",
            "γ-Exponential", "Matern"]
# all stationary kernels
const k_strings = vcat(iso_k_strings, "Cosine")
                #, "Spectral Mixture"]

r = 2*rand()
const iso_k_arr = [EQ(), Exp(), Delta(),
        Constant(r), RQ(r),
        γExp(r), Matern(r)]
const k_arr = vcat(iso_k_arr, Cosine(r))
        #, SM]

const tol = 1e-12

# TODO:
# test higher input dimensions
# test type stability

@testset "basic properties" begin
    T = Float64
    n = 16
    x = randn(T, n)
    Σ = zeros(T, (n, n))
    for (k, k_str) in zip(k_arr, k_strings)
        @testset "$k_str" begin
            Σ .= k.(x, permutedims(x))
            @test iscov(Σ, tol)
            @test isstationary(k, x)
        end
    end

    @testset "MaternP" begin
        # testing different inputs
        for p = 0:4
            k = MaternP(p)
            Σ .= k.(x, permutedims(x))
            @test iscov(Σ, tol)
            @test isstationary(k, x)
        end
        # testing constructor for invalid inputs
        @test_throws DomainError MaternP(-1)

        # can be constructed via Matern constructor with appropriate ν
        # we might not want to do that if we want to optimize ν
        for p = 0:4
            @test MaternP(Matern(p+1/2)) isa MaternP
        end

        # testing differentiability
        k = MaternP(1)
        @test derivative(k, 0) ≈ 0
    end

end

@testset "multidimensional inputs" begin
    n = 16
    Σ = zeros(n, n)
    for (k, k_str) in zip(k_arr, k_strings)
        if typeof(k) <: CovarianceFunctions.IsotropicKernel
            @testset "$k_str" begin
                for d = 2:3
                    x = [randn(d) for _ in 1:n]
                    K = k.(x, permutedims(x))
                    Σ = k.(x, permutedims(x))
                    @test iscov(Σ, tol)
                    @test isstationary(k, x)
                end
            end
        end
    end
end

@testset "kernel modifications" begin

    using CovarianceFunctions: Lengthscale
    l = exp(randn())
    for d = 1:3
        r = randn(d)
        for (k, k_str) in zip(iso_k_arr, iso_k_strings)
            kl = Lengthscale(k, l)
            @test kl(r) ≈ k(norm(r)/l)
            # @test typeof(kl) <: CovarianceFunctions.IsotropicKernel
            @test isstationary(kl)
        end
    end

    ########### Testing ARD
    d = 2
    k = CovarianceFunctions.EQ()
    l = ones(d)
    kl = CovarianceFunctions.ARD(k, l)
    x = randn(d)
    y = randn(d)

    @test kl(x, y) ≈ k(x, y)

    @test norm(x-y) ≈ kl.n(x-y) # this tests Metrics
    c = 2
    l = c^2*l
    kl = CovarianceFunctions.ARD(k, l)
    @test 1/c*norm(x-y) ≈ kl.n(x-y)

    l = exp.(randn(d))
    w = @. sqrt(1/l)
    kl = CovarianceFunctions.ARD(k, l)
    @test norm((x-y).*w) ≈ kl.n(x-y)
    xl = x .* w
    yl = y .* w
    @test kl(x, y) ≈ k(xl, yl)

    d = 3
    U = randn(1, d)
    n(x) = enorm(U'U, x)
    kl = CovarianceFunctions.Normed(k, n)
    x = randn(d)
    y = randn(d)
    @test kl(x, y) ≈ k(U*x, U*y)
    @test kl(x, y) ≈ k(sqrt((x-y)'*(U'U)*(x-y)))

    S = LowRank(U')
    kSLR = CovarianceFunctions.Energetic(k, S)
    @test kSLR(x, y) ≈ kl(x, y)

    # periodic CovarianceFunctions
    k = CovarianceFunctions.EQ()
    p = CovarianceFunctions.Periodic(k)
    x, y = randn(2)
    @test p(x, y) isa Real
    @test p(1+x) ≈ p(x) # 1 periodic
    @test p(x, 1+y) ≈ p(x, y)  # 1 periodic
    @test p(x+1, y-1) ≈ p(x, y)  # 1 periodic
    @test p(x+7, y-3) ≈ p(x, y)  # 1 periodic

    # TODO: write benchmarks
    # println(size(U*x))
    # println(size(x))
    # L = randn(d, 16)
    # @time k(L'x, L'y)
    # @time k(U*x, U*y)
    # @time kl(x, y)
    # using MyFactorizations: SymmetricLowRank
    # S = SymmetricLowRank(U)
    # kSLR = CovarianceFunctions.Subspace(k, S)

end # CovarianceFunctions modifications

end # TestStationary

# @testset "Rational Quadratic" begin
#     α = exp(randn())
#     k = RQ(α)
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
#     # test constructor for negative input
# end
#
# @testset "Exponential" begin
#     k = Exp()
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
# end
#
# @testset "γ-Exponential CovarianceFunctions" begin
#     γ = 2rand()
#     k = ΓExp(γ)
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
#     # test constructor for γ outside of [0,2]
# end
#
# @testset "δ CovarianceFunctions" begin
#     k = Delta()
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
# end
