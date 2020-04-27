module TestOptimization
# testing optimization of kernel hyper-parameters
using Test
using Kernel
using Kernel: parameters, nparameters

# testing if we can move between kernel structure and vector of kernel
# parameters with "parameter" and "similar"
@testset "vector isomorphism" begin

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

end # TestOptimization
