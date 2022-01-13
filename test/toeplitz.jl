module TestToeplitz
using LinearAlgebra
using CovarianceFunctions: durbin, durbin!, trench, trench!, levinson, levinson!
using ToeplitzMatrices
using Test

@testset "toeplitz" begin
    n = 16
    x = range(-1, 1, length = n+1)
    a_exp = @. exp(-abs(x[1]-x))
    a_rbf = @. exp(-abs(x[1]-x)^2) / 2 # this one can be ill-conditioned unless we add a diagonal, i.e. scale kernel down while keeping diagonal at 1
    a_rand = rand(n+1) / n # ensures diagonal dominance

    a_tuple = (a_rand, a_exp, a_rbf)
    for a in a_tuple
        a[1] = 1
        r = a[2:end]

        y = durbin(r)

        K = SymmetricToeplitz(vcat(1, r[1:end-1]))
        KM = Matrix(K)
        b = - (KM \ r)
        @test b ≈ y

        B = trench(r[1:end-1])
        invKM = inv(KM)
        @test B ≈ invKM

        a = a[1:end-1]
        r = a[2:end]
        b = randn(n)
        y = levinson(r, b)

        K = SymmetricToeplitz(vcat(1, r[1:end]))
        KM = Matrix(K)

        Kb = KM \ b
        @test Kb ≈ y
        @test Kb ≈ levinson(K, b)
    end
end

end # module
