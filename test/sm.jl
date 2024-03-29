module TestSpectralMixture
using CovarianceFunctions
using CovarianceFunctions: parameters, nparameters, isstationary
using Test
# IDEA: test on real data
# dataset:
# https://cdiac.ess-dive.lbl.gov/ftp/trends/co2/maunaloa.co2
@testset "spectral mixture kernel" begin
    n = 32
    x = sort!(randn(n))
    w = 1.
    μ = 0.
    l = 1.
    θ = [w, μ, l]
    k = CovarianceFunctions.Spectral(w, μ, l)
    @test k isa CovarianceFunctions.Product
    @test k(x[1], x[2]) isa Real
    @test isstationary(k)
    @test isstationary(k, x)

    ns = 3
    w = exp.(randn(ns))
    μ = exp.(randn(ns))
    l = exp.(randn(ns))
    θ = [w μ l]
    θ = θ'[:]
    sm = CovarianceFunctions.SpectralMixture(w, μ, l)
    @test isstationary(sm)
    @test isstationary(sm, x)


    # testing periodic
    # x = -1:.01:1
    # eq = CovarianceFunctions.EQ()
    # # eq = CovarianceFunctions.Lengthscale(eq, .5) # changes variation of periodic function without changing periodicity
    # per = CovarianceFunctions.Periodic(eq)
    # per = CovarianceFunctions.Lengthscale(per, .5) # changes periodicity
    # σ = .001
    # C = factorize(σ^2*I + CovarianceFunctions.gramian(per, x))
    # y = C.L * randn(length(x), 2)
    # using Plots
    # plot(x, y)
    # gui()
end


end # TestSpectralMixture
