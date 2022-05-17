module TestProperties
using Test
using CovarianceFunctions
using CovarianceFunctions: input_trait, DotProductInput, IsotropicInput, StationaryLinearFunctionalInput, GenericInput
using CovarianceFunctions: EQ, RQ, Exp, Dot, ExponentialDot, Poly, Line, Cos

using LinearAlgebra

@testset "properties" begin
    dot_kernels = [Dot(), Dot()^3, ExponentialDot(), Line(1.), Poly(5, 1.)]
    for k in dot_kernels
        @test input_trait(k) isa DotProductInput
    end

    iso_kernels = [EQ(), RQ(1.), Exp()]
    for k in iso_kernels
        @test input_trait(k) isa IsotropicInput
    end

    k = CovarianceFunctions.NeuralNetwork()
    @test input_trait(k) isa GenericInput

    # testing that constant kernels don't confuse the input_trait inference
    @test input_trait(1*EQ() + 1) isa IsotropicInput
    @test input_trait(1*EQ() + 2 + RQ(1.)*1) isa IsotropicInput

    @test input_trait(1*Dot() + 1) isa DotProductInput
    @test input_trait(1*Dot() + 2 + Dot()^2*1) isa DotProductInput

    w = randn()
    @test input_trait(1*Cos(w) + 1) isa StationaryLinearFunctionalInput
    @test input_trait(1*Cos(w) + 2 + Cos(w)^2*1) isa StationaryLinearFunctionalInput
end

end
