module TestStationary

using Test
using LinearAlgebra
using LinearAlgebraExtensions: LowRank

using Kernel
using Kernel: MercerKernel, StationaryKernel, isstationary, isisotropic
using Kernel: Constant, EQ, RQ, Exp, γExp, Delta, Cosine, MaternP, Matern#, SM
using Kernel: iscov

using Metrics
const euclidean = Metrics.EuclideanNorm()

####################### randomized stationarity tests ##########################
# WARNING this test only allows for euclidean isotropy ...
# does not matter for 1d tests
function Kernel.isisotropic(k::MercerKernel, x::AbstractVector)
    isiso = false
    if isstationary(k)
        isiso = true
        r = euclidean(x[1]-x[2])
        println(r)
        kxr = k(x_i/r, x_j/r)
        for x_i in x, x_j in x
            r = euclidean(x_i-x_j)
            val = k(x_i/r, x_j/r)
            if !(kxr ≈ val)
                isiso = false
            end
        end
    end
    return isiso
end

# tests if k is stationary on finite set of points x
# need to make this multidimensional
function Kernel.isstationary(k::MercerKernel, x::AbstractVector)
    n = length(x)
    d = length(x[1])
    is_stationary = true
    for i in 1:n, j in 1:n
        ε = eltype(x) <: AbstractArray ? randn(d) : randn()
        iseq = k(x[i], x[j]) ≈ k(x[i]+ε, x[j]+ε)
        if !iseq
            println(i ,j)
            println(x[i], x[j])
            println(k(x[i], x[j]) - k(x[i]+ε, x[j]+ε))
            is_stationary = false
            break
        end
    end
    K = typeof(k)
    if !is_stationary && (K <: StationaryKernel)
        println("Covariance function " * string(K) * " is non-stationary but is subtype of StationaryKernel.")
        return false
    end

    # if the kernel seems stationary but isn't a subtype of stationary kernel, suggest making it one
    if is_stationary && !(K <: StationaryKernel)
        println("Covariance function " * string(K) * " seems to be stationary. Consider making it a subtype of StationaryKernel.")
        return false
    end
    return is_stationary
end

const k_strings = ["Exponentiated Quadratic", "Exponential", "δ",
            "Constant", "Rational Quadratic",
            "γ-Exponential", "Cosine",
            "Matern"]#, "Spectral Mixture"]

r = 2*rand()
const k_arr = [EQ(), Exp(), Delta(),
        Constant(r), RQ(r),
        γExp(r), Cosine(r),
        Matern(r)]#, SM]

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
        for p = 1:4
            k = MaternP(p)
            Σ .= k.(x, permutedims(x))
            @test iscov(Σ, tol)
            @test isstationary(k, x)
        end
        # testing constructor for invalid inputs
        @test_throws DomainError MaternP(-1)

        # can be constructed via Matern constructor with appropriate ν
        # we might not want to do that if we want to optimize ν
        for p = 1:4
            @test typeof(MaternP(Matern(p+1/2)))<:MaternP
        end
    end

end

@testset "multidimensional inputs" begin
    n = 16
    Σ = zeros(n, n)
    for (k, k_str) in zip(k_arr, k_strings)
        if typeof(k) <: Kernel.IsotropicKernel
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

    # k_strings = ["Exponentiated Quadratic", "Exponential", "δ",
    #             "Constant", "Rational Quadratic",
    #             "γ-Exponential",
    #             "Matern"]#, "Spectral Mixture"]
    #
    # r = 2*rand()
    # k_arr = [EQ(), Exp(), Delta(),
    #         Constant(r), RQ(r),
    #         γExp(r),
    #         Matern(r)]#, SM]

    using Kernel: Lengthscale
    l = exp(randn())
    for d = 1:3
        r = randn(d)
        for (k, k_str) in zip(k_arr, k_strings)
            kl = Lengthscale(k, l)
            @test kl(r) ≈ k(euclidean(r)/l)
            # @test typeof(kl) <: Kernel.IsotropicKernel
            @test isstationary(kl)
        end
    end

    using Kernel: Normed
    using Metrics
    using Metrics: Energetic

    # n = Metrics.norm(Energetic()
    ########### Testing ARD
    d = 2
    k = Kernel.EQ()
    l = ones(d)
    kl = Kernel.ARD(k, l)
    x = randn(d)
    y = randn(d)

    @test kl(x, y) ≈ k(x, y)

    @test norm(x-y) ≈ kl.n(x-y) # this tests Metrics
    c = 2
    l = c^2*l
    kl = Kernel.ARD(k, l)
    @test 1/c*norm(x-y) ≈ kl.n(x-y)

    l = exp.(randn(d))
    w = @. sqrt(1/l)
    kl = Kernel.ARD(k, l)
    @test norm((x-y).*w) ≈ kl.n(x-y)
    xl = x .* w
    yl = y .* w
    @test kl(x, y) ≈ k(xl, yl)

    d = 3
    U = randn(1, d)
    n = Metrics.EnergeticNorm(U'U)
    # kl = Kernel.Normed(k, n)
    kl = Kernel.Energetic(k, U'U)
    x = randn(d)
    y = randn(d)
    @test kl(x, y) ≈ k(U*x, U*y)
    @test kl(x, y) ≈ k(sqrt((x-y)'*(U'U)*(x-y)))

    S = LowRank(U')
    kSLR = Kernel.Energetic(k, S)
    @test kSLR(x, y) ≈ kl(x, y)

    # TODO: Learn how to write benchmarks
    # println(size(U*x))
    # println(size(x))
    # L = randn(d, 16)
    # @time k(L'x, L'y)
    # @time k(U*x, U*y)
    # @time kl(x, y)
    # using MyFactorizations: SymmetricLowRank
    # S = SymmetricLowRank(U)
    # kSLR = Kernel.Subspace(k, S)

end # kernel modifications

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
# @testset "γ-Exponential Kernel" begin
#     γ = 2rand()
#     k = ΓExp(γ)
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
#     # test constructor for γ outside of [0,2]
# end
#
# @testset "δ Kernel" begin
#     k = Delta()
#     Σ .= k.(x, permutedims(x))
#     @test iscov(Σ)
#     @test isstationary(k, x)
# end
