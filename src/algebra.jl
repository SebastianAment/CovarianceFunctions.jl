############################# kernel algebra ###################################
# IDEA: separable sum gramian
# IDEA: (Separable) Sum and Product could be one definition with meta programming
################################ Product #######################################
# TODO: constructors which merge products and sums
struct Product{T, AT<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel{T}
    args::AT
    function Product(k::Tuple{Vararg{AbstractKernel}})
        T = promote_type(eltype.(k)...)
        new{T, typeof(k)}(k)
    end
end
@functor Product
(P::Product)(τ) = prod(k->k(τ), P.args) # TODO could check for isotropy here
(P::Product)(x, y) = prod(k->k(x, y), P.args)
# (P::Product)(x, y) = isstationary(P) ? P(difference(x, y)) : prod(k->k(x, y), P.args)
Product(k::AbstractKernel...) = Product(k)
Product(k::AbstractVector{<:AbstractKernel}) = Product(k...)
Base.prod(k::AbstractVector{<:AbstractKernel}) = Product(k)

Base.:*(k::AbstractKernel...) = Product(k)
Base.:*(c::Number, k::AbstractKernel) = Constant(c) * k
Base.:*(k::AbstractKernel, c::Number) = Constant(c) * k

################################### Sum ########################################
struct Sum{T, AT<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel{T}
    args::AT
    function Sum(k::Tuple{Vararg{AbstractKernel}})
        T = promote_type(eltype.(k)...)
        new{T, typeof(k)}(k)
    end
end
@functor Sum
(S::Sum)(τ) = sum(k->k(τ), S.args) # should only be called if S is stationary
(S::Sum)(x, y) = sum(k->k(x, y), S.args)
# (S::Sum)(τ) = isstationary(S) ? sum(k->k(τ), S.args) : error("One argument evaluation not possible for non-stationary kernel")
# (S::Sum)(x, y) = isstationary(S) ? S(difference(x, y)) : sum(k->k(x, y), S.args)
Sum(k::AbstractKernel...) = Sum(k)
Sum(k::AbstractVector{<:AbstractKernel}) = Sum(k...)
Base.sum(k::AbstractVector{<:AbstractKernel}) = Sum(k)

Base.:+(k::AbstractKernel...) = Sum(k)
Base.:+(k::AbstractKernel, c::Number) = k + Constant(c)
Base.:+(c::Number, k::AbstractKernel) = k + Constant(c)

################################## Power #######################################
struct Power{T, K<:AbstractKernel{T}, PT} <: AbstractKernel{T}
    k::K
    p::PT
end
@functor Power
(P::Power)(τ) = P.k(τ)^P.p
(P::Power)(x, y) = P.k(x, y)^P.p
Base.:^(k::AbstractKernel, p::Number) = Power(k, p)

############################ Separable Product #################################
# product kernel, but separately evaluates component kernels on different parts of the input
struct SeparableProduct{T, K} <: AbstractKernel{T}
    args::K # kernel for input covariances
end
@functor SeparableProduct
SeparableProduct(k...) = SeparableProduct(k)
function SeparableProduct(k::Union{Tuple, AbstractVector})
    T = promote_type(eltype.(k)...)
    SeparableProduct{T, typeof(k)}(k)
end
# both x and y have to be vectors of inputs to individual kernels
# could also consist of tuples ... so restricting to AbstractVector might not be good
function (k::SeparableProduct)(x::AbstractVector, y::AbstractVector)
    d = checklength(x, y)
    length(k.args) == d || throw(DimensionMismatch("SeparableProduct needs d = $d kernels but has r = $(length(k.args))"))
    val = one(gramian_eltype(k, x, y))
    @inbounds @simd for i in eachindex(k.args)
        ki = k.args[i]
        val *= ki(x[i], y[i])
    end
    return val
end

# if we had kernel input type, could compare with eltype(X)
function gramian(k::SeparableProduct, X::LazyGrid, Y::LazyGrid)
    length(X.args) == length(Y.args) || throw(DimensionMismatch("length(X.args) = $(length(X.args)) ≠ $(length(Y.args)) = length(Y.args)"))
    length(k.args) == length(X.args) || throw(DimensionMismatch("SeparableProduct needs d = $(length(X.args)) kernels but has r = $(length(k.args))"))
    kronecker((gramian(kxy...) for kxy in zip(k.args, X.args, Y.args))...)
end
# IDEA: if points are not on a grid, can still evaluate dimensions separately,
# and take elementwise product. can lead to efficiency gains if constituent
# matrices are of low rank (see SKIP paper)
############################### Separable Sum ##################################
# what about separable sums? do they give rise to kronecker sums? yes!
# useful for "Additive Gaussian Processes" - Duvenaud 2011
# https://papers.nips.cc/paper/2011/file/4c5bde74a8f110656874902f07378009-Paper.pdf
# IDEA: could introduce have AdditiveKernel with "order" argument, that adds higher order interactions
struct SeparableSum{T, K} <: AbstractKernel{T}
    args::K # kernel for input covariances
end
@functor SeparableSum

SeparableSum(k...) = SeparableSum(k)
function SeparableSum(k::Union{Tuple, AbstractVector})
    T = promote_type(eltype.(k)...)
    SeparableSum{T, typeof(k)}(k)
end

function (k::SeparableSum)(x::AbstractVector, y::AbstractVector)
    d = checklength(x, y)
    length(k.args) == d || throw(DimensionMismatch("SeparableProduct needs d = $d kernels but has r = $(length(k.args))"))
    val = one(gramian_eltype(k, x, y))
    @inbounds @simd for i in eachindex(k.args)
        ki = k.args[i]
        val += ki(x[i], y[i])
    end
    return val
end

# IDEA: does gramian have special structure, like kronecker sum?
# function gramian(k::SeparableSum, X::LazyGrid, Y::LazyGrid)
#     # ⊕((gramian(kxy...) for kxy in zip(K.args, X.args, Y.args))...)
#     x_lengths = [length(x) for x in X.args]
#     y_lengths = [length(y) for y in Y.args]
#     G = zeros(length(X), length(Y))
#     for i in eachindex(k.args)
#         ki, xi, yi = k.args[i], X.args[i], Y.args[i]
#         G .+= kron(gramian(ki, xi, yi))
#     end
#     return G
# end

# convenient constructor
# e.g. separable(*, k1, k2)
separable(::typeof(*), k...) = SeparableProduct(k)
separable(::typeof(+), k...) = SeparableSum(k)
# d-separable product of k
separable(::typeof(^), k::AbstractKernel, d::Int) = SeparableProduct(Fill(k, d))
