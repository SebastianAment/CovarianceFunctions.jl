module TestOptimization
# testing helpers for optimization of CovarianceFunctions hyper-parameters
using Test
using CovarianceFunctions
using CovarianceFunctions: parameters, nparameters
using LinearAlgebra

# testing if we can move between CovarianceFunctions structure and vector of CovarianceFunctions
# parameters with "parameter" and "similar"
@testset "CovarianceFunctions-vector isomorphism" begin

    # parameter test
    eq = CovarianceFunctions.EQ()
    α = exp(randn())
    rq = CovarianceFunctions.RQ(α)
    @test parameters(eq) == []
    @test parameters(rq) == [α]
    @test parameters(eq + rq) == [α]
    @test parameters(eq * rq) == [α]
    @test parameters(rq^2) == [α]
    ν = exp(randn())
    matern = CovarianceFunctions.Matern(ν)
    @test parameters(matern * rq) == [ν, α]
    @test nparameters(matern + rq) == 2

    # similar test
    @test similar(eq, []) isa CovarianceFunctions.EQ
    @test similar(rq, α) isa CovarianceFunctions.RQ
    @test similar(eq+rq, α) isa typeof(eq+rq)
    s = matern*rq
    @test similar(s, [ν, α]) isa typeof(s)
    @test parameters(similar(s, [ν, α])) == parameters(s)
    @test similar(eq^2, []) == eq^2

    sym = CovarianceFunctions.SymmetricKernel(eq, 0.)
    @test similar(sym, randn(1)) isa typeof(sym)
    @test parameters(sym)[] == 0.
    @test nparameters(sym) == 1
end

end # TestOptimization
