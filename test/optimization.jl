module TestOptimization
# testing optimization of kernel hyper-parameters
using Test
using Kernel
using Kernel: parameters, nparameters
using LinearAlgebra

function plot_posterior(k, xs, x, y, σ)
    K = Kernel.gramian(k, x)
    Ks = Kernel.gramian(k, xs, x)
    Σ = cholesky(σ^2*I+K)
    plot!(xs, Ks * (Σ\y), ribbon = sqrt.(diag(Kernel.gramian(k, xs) - Ks * (Σ \ Ks'))))
end

# testing if we can move between kernel structure and vector of kernel
# parameters with "parameter" and "similar"
@testset "kernel-vector isomorphism" begin

    # parameter test
    eq = Kernel.EQ()
    α = exp(randn())
    rq = Kernel.RQ(α)
    @test parameters(eq) == []
    @test parameters(rq) == [α]
    @test parameters(eq + rq) == [α]
    @test parameters(eq * rq) == [α]
    @test parameters(rq^2) == [α]
    ν = exp(randn())
    matern = Kernel.Matern(ν)
    @test parameters(matern * rq) == [ν, α]
    @test nparameters(matern + rq) == 2

    # similar test
    @test similar(eq, []) isa Kernel.EQ
    @test similar(rq, α) isa Kernel.RQ
    @test similar(eq+rq, α) isa typeof(eq+rq)
    s = matern*rq
    @test similar(s, [ν, α]) isa typeof(s)
    @test parameters(similar(s, [ν, α])) == parameters(s)
    @test similar(eq^2, []) == eq^2

    sym = Kernel.Symmetric(eq, 0.)
    @test similar(sym, randn(1)) isa typeof(sym)
    @test parameters(sym)[] == 0.
    @test nparameters(sym) == 1
end

@testset "optimization" begin
    using Optimization: value, value_direction
    using Kernel: KernelGradient

    # using NormalDistributions
    include("nlml.jl")
    l = .2
    k = Kernel.Lengthscale(Kernel.EQ(), l)
    n = 32
    x = sort!(randn(n))
    K = Kernel.gramian(k, x)
    σ = .2
    C = cholesky(σ^2*I + K)
    y = C.L * randn(n)
    # @time dK = Kernel.derivative_matrix(K, 1)
    # @time dK = Kernel.derivative!_matrix(K, 1)
    # @time Matrix(K)

    D = KernelGradient(k, x, y, σ)
    θ = parameters(k)
    Kernel.parameterize!(θ)
    @test value(D, θ) isa Real
    @test value(D, θ) ≈ nlml(σ^2*I + k.(x, x'), y)

    val, dir = value_direction(D, θ)
    @test val isa Real
    @test dir isa AbstractVector
    @test length(dir) == length(θ)

    θ = [2.]
    k = similar(k, θ)
    σ2 = 1.

    k2, σ2 = Kernel.optimize(k, x, y, σ2)

    @test isapprox(σ, σ2, rtol = .5)
    @test isapprox([l], parameters(k2), rtol = .5)
    doplot = true
    if doplot
        using Plots
        plotly()
        scatter(x, y)
        xs = -3:.01:3
        plot_posterior(k, xs, x, y, σ)
        plot_posterior(k2, xs, x, y, σ2)
        gui()
    end
end

end # TestOptimization

# gradient test
# using ForwardDiff
# function f(θ)
#     println("in f")
#     println(θ)
#     θ = exp.(θ)
#     σ = θ[1]
#     println("σ = $σ")
#     kn = similar(k, θ[2:end])
#     K = Kernel.gramian(kn, x)
#     Σ = cholesky(σ^2*I+K)
#     nlml(Σ, y)
# end
# grad = ForwardDiff.gradient(f, log.(vcat(σ, l)))
# println("printing")
# println(dir)
# println(-grad)

# using ForwardDiff
# function g(log_θ)
#     println("in g")
#     println(log_θ)
#     θ = exp.(log_θ)
#     σ = θ[1]
#     println("σ = $σ")
#     Σ = cholesky(σ^2*I+K)
#     nlml(Σ, y)
# end
# grad = ForwardDiff.gradient(g, log.([σ]))
# println("printing")
# println(dir)
# println(-grad)
