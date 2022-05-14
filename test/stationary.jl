module TestStationary

using Test
using LinearAlgebra
# using LinearAlgebraExtensions: LowRank

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
    x = randn(T, n) # IDEA: include higher-dimensional test cases
    Σ = zeros(T, (n, n))
    r² = sum(abs2, x[1] - x[2])
    for k in iso_k_arr
        @test k(x[1], x[2]) ≈ k(r²)
    end
    for (k, k_str) in zip(k_arr, k_strings)
        @testset "$k_str" begin
            Σ .= k.(x, permutedims(x))
            @test iscov(Σ, tol)
            @test isstationary(k, x)
        end
    end

    @testset "Matern" begin
        # testing different inputs
        for p = 0:8
            k = MaternP(p)
            Σ .= k.(x, permutedims(x))
            @test iscov(Σ, tol)
            @test isstationary(k, x)
        end

        # testing against control implementation
        # r² = x.^2
        r² = 10.0.^(1:16) * eps()
        p = 0
        k = MaternP(p)
        @test k.(r²) ≈ MaternP.(r², p)
        for p = 2:3
            k = MaternP(p)
            @test k.(0) ≈ MaternP.(0, p) # evaluation at zero
            @test k.(r²) ≈ MaternP.(r², p) # for small numbers up to O(1)
            # testing differentiability
            # k_naive = z->MaternP(z, p)
            k_naive = z->Matern(z, p + 1/2)
            dk_taylor = derivative.((k,), r²)
            dk_naive = derivative.((k_naive,), r²)
            #
            @test isapprox(dk_taylor, dk_naive, atol = 1e-6)

            # second order problem: naïve implementation is not as accurate anymore
            dk_taylor = derivative.((z->derivative(k, z),), r²)
            dk_naive = derivative.((z->derivative(k_naive, z),), r²)
            @test isapprox(dk_taylor, dk_naive, atol = 1e-5)
        end


        # testing constructor for invalid inputs
        @test_throws DomainError MaternP(-1)

        # can be constructed via Matern constructor with appropriate ν
        # we might not want to do that if we want to optimize ν
        for p = 0:4
            @test MaternP(Matern(p+1/2)) isa MaternP
        end

        # k = Matern(1.5)
        # @test derivative(k, 0) ≈ ... # this needs to be fixed with taylor expansion
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
            @test kl(r) ≈ k(sum(abs2, r)/l^2)
            @test typeof(kl) <: CovarianceFunctions.IsotropicKernel
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

    @test sum(abs2, x-y) ≈ kl.n²(x-y) # this tests Metrics
    c = 2
    l = c^2*l
    kl = CovarianceFunctions.ARD(k, l)
    @test 1/c^2*sum(abs2, x-y) ≈ kl.n²(x-y)

    l = exp.(randn(d))
    w = @. sqrt(1/l)
    kl = CovarianceFunctions.ARD(k, l)
    @test sum(abs2, (x-y).*w) ≈ kl.n²(x-y)
    xl = x .* w
    yl = y .* w
    @test kl(x, y) ≈ k(xl, yl)

    d = 3
    U = randn(1, d)
    n²(x) = CovarianceFunctions.enorm2(U'U, x)
    kl = CovarianceFunctions.Normed(k, n²)
    x = randn(d)
    y = randn(d)
    @test kl(x, y) ≈ k(U*x, U*y)
    @test kl(x, y) ≈ k((x-y)'*(U'U)*(x-y))

    S = U'U # LowRank(U')
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
