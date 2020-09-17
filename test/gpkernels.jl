module TestGPKernels
using Test
using Kernel
using Kernel: ConditionalKernel, conditional, posterior, gramian
using LinearAlgebra

@testset "gpkernels" begin
    # create dataset
    n = 4
    x = 2randn(n)
    sort!(x)

    # pick base kernel
    k = Kernel.EQ()

    tol = 1e2eps(Float64)
    ck = conditional(k, x)
    @test all(ck.(x, x) .< tol) # uncertainty at interpolation points is zero

    pk = posterior(k, .1, x)
    @test !all(pk.(x, x) .< tol) # uncertainty at interpolation points is zero

    # xs = randn(n) # test points
    xs = 2minimum(x):.05:2maximum(x)
    for k in (ck, pk)
        K = Symmetric(Matrix(gramian(k, xs)))
        @test minimum(eigvals(K)) ≥ -tol
    end
    
    # using Plots
    # plotly()
    # surface(Matrix(CK))
    # gui()


    # xs = 2minimum(x):.01:2maximum(x)
    # plot(xs, ck.(xs, xs))
    # plot!(xs, pk.(xs, xs))
    # scatter!(x, zero(x))
    # gui()
end

end # TestGPKernels
