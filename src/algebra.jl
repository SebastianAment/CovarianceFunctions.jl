############################# kernel algebra ###################################
# NOTE: output type inference of product, sum, and power not supported for
# user-defined kernels unless Base.eltype is defined for them
################################ Product #######################################
struct Product{T, AT<:Union{Tuple, AbstractVector}, IT} <: AbstractKernel{T}
    args::AT
    input_trait::IT # could keep track of the overall input trait
end
@functor Product
function Product(k::Union{Tuple, AbstractVector})
    T = promote_type(eltype.(k)...)
    IT = sum_and_product_input_trait(k)
    Product{T, typeof(k), typeof(IT)}(k, IT)
end
Product(k...) = Product(k)
(P::Product)(τ) = prod(k->k(τ), P.args) # IDEA could check for isotropy here
(P::Product)(x, y) = prod(k->k(x, y), P.args)
# (P::Product)(x, y) = isisotropic(P) ? P(difference(x, y)) : prod(k->k(x, y), P.args)
Product(k::AbstractKernel...) = Product(k)
Product(k::AbstractVector{<:AbstractKernel}) = Product(k...)
Base.prod(k::AbstractVector{<:AbstractKernel}) = Product(k)

Base.:*(k::AbstractKernel...) = Product(k)
Base.:*(c::Number, k::AbstractKernel) = Constant(c) * k
Base.:*(k::AbstractKernel, c::Number) = Constant(c) * k

################################### Sum ########################################
struct Sum{T, AT<:Union{Tuple, AbstractVector}, IT} <: AbstractKernel{T}
    args::AT
    input_trait::IT # could keep track of the overall input trait
end
@functor Sum
function Sum(k::Union{Tuple, AbstractVector})
    T = promote_type(eltype.(k)...)
    IT = sum_and_product_input_trait(k)
    Sum{T, typeof(k), typeof(IT)}(k, IT)
end
Sum(k...) = Sum(k)
(S::Sum)(τ) = sum(k->k(τ), S.args) # should only be called if S is stationary
(S::Sum)(x, y) = sum(k->k(x, y), S.args)
# (S::Sum)(τ) = isstationary(S) ? sum(k->k(τ), S.args) : error("One argument evaluation not possible for non-stationary kernel")
# (S::Sum)(x, y) = isstationary(S) ? S(difference(x, y)) : sum(k->k(x, y), S.args)
Base.sum(k::AbstractVector{<:AbstractKernel}) = Sum(k)

Base.:+(k::AbstractKernel...) = Sum(k)
Base.:+(k::AbstractKernel, c::Number) = k + Constant(c)
Base.:+(c::Number, k::AbstractKernel) = k + Constant(c)

################################## Power #######################################
struct Power{T, K, IT} <: AbstractKernel{T}
    k::K
    p::Int
    input_trait::IT # could keep track of the overall input trait
end
@functor Power
function Power(k, p::Int)
    T = promote_type(eltype(k))
    IT = input_trait(k)
    Power{T, typeof(k), typeof(IT)}(k, p, IT)
end
(P::Power)(τ) = P.k(τ)^P.p
(P::Power)(x, y) = P.k(x, y)^P.p
Base.:^(k::AbstractKernel, p::Number) = Power(k, p)

############################ Separable Product #################################
# product kernel, but separately evaluates component kernels on different parts of the input
# NOTE: input_trait(::SeparableProduct) defaults to GenericInput
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
# NOTE: input_trait(::SeparableSum) defaults to GenericInput
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
