module TestProperties
using Test
using CovarianceFunctions
using CovarianceFunctions: input_trait, DotProductInput, IsotropicInput, GenericInput
using CovarianceFunctions: EQ, RQ, Exp, Dot, Poly, Line

using LinearAlgebra

@testset "properties" begin
    dot_kernels = [Dot(), Dot()^3] # , Line(1.), Poly(5, 1.)] # TODO: take care of constants
    for k in dot_kernels
        @test input_trait(k) isa DotProductInput
    end

    iso_kernels = [EQ(), RQ(1.), Exp()]
    for k in iso_kernels
        @test input_trait(k) isa IsotropicInput
    end

    k = CovarianceFunctions.NeuralNetwork()
    @test input_trait(k) isa GenericInput
end

end
