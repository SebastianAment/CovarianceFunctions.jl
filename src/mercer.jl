########################## non-stationary kernels ##############################
########################### dot product kernel #################################
struct Dot{T} <: MercerKernel{T} end
Dot() = Dot{Float64}()
(k::Dot)(x, y) = dot(x, y)

# all of the kernels below can be defined as a composition of Dot(), with other kernel modifications
Line(σ::Real = 0.) = Dot() + σ
Polynomial(d::Int, σ::Real = 0.) = Line(σ)^d
const Poly = Polynomial

############################# Matrix kernel ####################################
# this could be discrete input kernel, as opposed to a matrix valued kernel
struct MatrixKernel{T, AT<:AbstractMatrix{T}} <: MercerKernel{T}
    A::AT
end
(k::MatrixKernel)(x::Integer, y::Integer) = k.A[i,j]

############################ Brownian kernel ###################################
# technically stationary increments ...
struct Brownian{T} <: MercerKernel{T} end
# squared euclidean distance of x, y in the space (cos(x), sin(x))
# since (cos(x) - cos(y))^2 + (sin(x) - sin(y))^2 = 4*sin((x-y)/2)^2
(k::Brownian)(x::Real, y::Real) = min(x, y)

# NN(σ) = π/2 * (asin ∘ (Line(σ) / x->√(1+dot(x,x))))
# written with less sleek syntax: linear + normalized + asin (transform)
# NN(σ) = Transform(VerticalRescaling(Line(σ), x->1/√(1+dot(x,x))), asin)
# finite basis function (linear regression) kernel,
struct FiniteBasis{T, B<:Union{AbstractVector, Tuple}} <: MercerKernel{T}
    basis::B # tuple of vector of functions
    function FiniteBasis{T}(basis) where {T}
        length(basis) ≥ 1 || throw("basis is empty: length(basis) = $(length(basis))")
        new{T, typeof(basis)}(basis)
    end
end
# FiniteBasis{T}(basis) where T = FiniteBasis{T, typeof(basis)}(basis)
FiniteBasis(basis) = FiniteBasis{Float64}(basis)
function (k::FiniteBasis)(x, y)
    v = zero(eltype(k))
    for b in k.basis
        v += b(x)*b(y)
    end
    return v
end
# TODO: have in-place versions?
function basis(k::FiniteBasis, x::AbstractVector)
    U = zeros(eltype(k), (length(x), length(k.basis)))
    for (i, b) in enumerate(k.basis) # could use threads here
        @. U[:, i] = b(x)
    end
    U
end
using LinearAlgebraExtensions: LowRank
function gramian(k::FiniteBasis, x::AbstractVector, y::AbstractVector)
    r = length(k.basis)
    if length(x) > r && length(y) > r # condition on low-rankness
        U = basis(k, x)
        V = x === y ? U : basis(k, y)
        return LowRank(U, V')
    else
        return Gramian(k, x, y) # should these be dense?
    end
end

########################### polynomial kernel ##################################
# is a composition of Dot + Power kernels
# struct Poly{T} <: MercerKernel{T}
#     σ::T
#     d::Int
#     function Poly{T}(σ, d) where T
#         if d < 0
#             error("degree negative")
#         elseif σ < 0
#             error("σ negative")
#         end
#         new(σ, d)
#     end
# end
#
# Poly{T}(d::Int) where T = Poly(T(0), d)
# Poly(d::Int) = Poly{Float64}(d)
#
# # as a function of the inner product xy (could be generalized)
# (k::Poly)(xy::Real) = (xy + k.σ)^k.d
# (k::Poly)(x::RTV, y::RTV) = k(dot(x, y))
# again this is a composition of simpler kernels: Power(Sum(Dot, Const), p)
# Poly(d, σ) = (Dot() + σ)^d

########################### neural network kernel ##############################
struct NeuralNetwork{T} <: MercerKernel{T}
    σ::T # have to restrict to be positive
    # TODO: inner constructor
end
const NN = NeuralNetwork

# NN{T}(σ::T = zero(T)) where {T} = NN(σ)
NN() = NN{Float64}(0.)

function (k::NN)(x, y)
    l = Line(k.σ)
    2/π * asin(l(x,y) / sqrt((1+l(x,x)) * (1+l(y,y))))
end

########################## more future additions ###############################
# TODO: Deep neural network kernel, convolutional NN kernel,
# Gibbs non-stationary construction etc.


############## kernel on subselection of input coordinates #####################
# struct Subselect{T, K<:MercerKernel{T}, TU<:Tuple}
#     k::K
#     ind::TU
# end
#
# function (k::Subselect{T})(x::T, y::T) where {T}
#     return k.k(x[ind], y[ind])
# end

# deprecated
# @inline _line(x, y, σ) = 2dot(x,y) + σ
# @inline _NN(x, y, σ) =  2/π * asin((_line(x,y,σ)) / sqrt((1+_line(x,x,σ)) * (1+_line(y,y,σ))))
# (k::NN)(x::T, y::T) where {T<:RTV} = _NN(x, y, k.σ)
