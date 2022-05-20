module TestDifferentiableGivens
# test
using Test
using LinearAlgebra
using LinearAlgebra: givensAlgorithm
using ForwardDiff: gradient, derivative

i = 3
g = x->givensAlgorithm(x, 2x)[i]
nexp = 16
@testset "differentiating givensAlgorithm" begin
    for _ in 1:nexp
        x = randn()
        gx = g(x)
        h = 1e-4
        y = x + h
        gy = g(y)
        gd_fd = (gy - gx) / h

        gd = derivative(g, x)

        tol = 1e-4
        @test abs(gd_fd - gd) < tol
    end
end

end
