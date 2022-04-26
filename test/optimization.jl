module TestOptimization
# testing helpers for optimization of CovarianceFunctions hyper-parameters
using Test
using CovarianceFunctions
using CovarianceFunctions: EQ, RQ, Lengthscale
using LinearAlgebra
using Flux

# testing taking gradients of kernels and incrementing parameters with them
@testset "optimization" begin
    l = [1/2]
    k = Lengthscale(EQ(), l)
    p = Flux.params(k)
    x, y = randn(), randn()
    g = gradient(()->k(x, y), p) # should we use @forward?
    @test g[l][1] isa Real
end

end # TestOptimization
