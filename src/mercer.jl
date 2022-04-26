########################## non-stationary kernels ##############################
abstract type DotProductKernel{T} <: MercerKernel{T} end
(k::DotProductKernel)(x, y) = k(dot(x, y))

########################### dot product kernel #################################
struct Dot{T} <: DotProductKernel{T} end
Dot() = Dot{Union{}}()
@functor Dot
@inline (k::Dot)(d::Number) = d

# all of the kernels below can be defined as a composition of Dot(), with other kernel modifications
Line(σ::Real = 0.) = Dot() + σ
Polynomial(d::Int, σ::Real = 0.) = Line(σ)^d
const Poly = Polynomial

# exponential inner product kernel
# can be interpreted as infinite weighted combination of polynomial kernels
# WARNING: not well behaved for large inner products
struct ExponentialDot{T} <: DotProductKernel{T} end
ExponentialDot() = ExponentialDot{Union{}}()
@functor ExponentialDot
(k::ExponentialDot)(d::Number) = exp(d)

############################# Matrix kernel ####################################
# this could be discrete input kernel, as opposed to a matrix valued kernel
struct MatrixKernel{T, AT<:AbstractMatrix{T}} <: MercerKernel{T}
    A::AT
end
@functor MatrixKernel
(k::MatrixKernel)(x::Integer, y::Integer) = k.A[i,j]

############################ Brownian kernel ###################################
# has stationary increments
struct Brownian{T} <: MercerKernel{T} end
@functor Brownian
Brownian() = Brownian{Union{}}()
(k::Brownian)(x::Real, y::Real) = min(x, y)

######################### Finite Basis kernel ##################################
# finite basis function (linear regression) kernel,
struct FiniteBasis{T, B<:Union{AbstractVector, Tuple}} <: MercerKernel{T}
    basis::B # tuple of vector of functions
    function FiniteBasis{T}(basis) where {T}
        length(basis) ≥ 1 || throw("basis is empty: length(basis) = $(length(basis))")
        new{T, typeof(basis)}(basis)
    end
end
@functor FiniteBasis
FiniteBasis(basis) = FiniteBasis{Float64}(basis)
(k::FiniteBasis)(x, y) = sum(b->b(x)*b(y), k.basis)

# TODO: have in-place versions?
function basis(k::FiniteBasis, x::AbstractVector)
    U = zeros(eltype(k), (length(x), length(k.basis)))
    for (i, b) in enumerate(k.basis) # could use threads here
        @. U[:, i] = b(x)
    end
    U
end

function gramian(k::FiniteBasis, x::AbstractVector, y::AbstractVector)
    r = length(k.basis)
    if length(x) > r && length(y) > r # condition on low-rankness
        U = basis(k, x)
        V = x === y ? U : basis(k, y)
        return LazyMatrixProduct(U, V')
    else
        return Gramian(k, x, y) # should these be dense?
    end
end

########################### neural network kernel ##############################
struct NeuralNetwork{T} <: MercerKernel{T}
    σ::T # have to restrict to be positive
    # TODO: inner constructor
end
const NN = NeuralNetwork
@functor NN
# NN{T}(σ::T = one(T)) where {T} = NN(σ)
NN() = NN{Float64}(0.)

function (k::NN)(x, y)
    l = Line(k.σ)
    2/π * asin(l(x, y) / sqrt((1 + l(x, x)) * (1 + l(y, y))))
end
# NN(σ) = π/2 * (asin ∘ (Line(σ) / x->√(1+dot(x,x))))
# written with less sleek syntax: linear + normalized + asin (transform)
# NN(σ) = Transform(VerticalRescaling(Line(σ), x->1/√(1+dot(x,x))), asin)

########################## more future additions ###############################
# IDEA: Deep neural network kernel, convolutional NN kernel,
# Gibbs non-stationary construction etc.
