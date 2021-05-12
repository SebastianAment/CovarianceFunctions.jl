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
(P::Product)(τ) = prod(k->k(τ), P.args) # TODO could check for isotropy here
(P::Product)(x, y) = prod(k->k(x, y), P.args)
# (P::Product)(x, y) = isstationary(P) ? P(difference(x, y)) : prod(k->k(x, y), P.args)
Product(k::AbstractKernel...) = Product(k)
Product(k::AbstractVector{<:AbstractKernel}) = Product(k...)
Base.prod(k::AbstractVector{<:AbstractKernel}) = Product(k)

Base.:*(k::AbstractKernel...) = Product(k)
Base.:*(c::Number, k::AbstractKernel) = Constant(c) * k
Base.:*(k::AbstractKernel, c::Number) = Constant(c) * k

parameters(k::Product) = vcat(parameters.(k.args)...)
nparameters(k::Product) = sum(nparameters, k.args)
function Base.similar(k::Product, θ::AbstractVector)
    return Product(_similar_helper(k, θ))
end

################################### Sum ########################################
struct Sum{T, AT<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel{T}
    args::AT
    function Sum(k::Tuple{Vararg{AbstractKernel}})
        T = promote_type(eltype.(k)...)
        new{T, typeof(k)}(k)
    end
end
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

parameters(k::Sum) = vcat(parameters.(k.args)...)
nparameters(k::Sum) = sum(nparameters, k.args)

# constructs similar object to k, but with different values θ
# overloading similar from Base
function Base.similar(k::Sum, θ::AbstractVector)
    return Sum(_similar_helper(k, θ))
end

################################## Power #######################################
struct Power{T, K<:AbstractKernel{T}} <: AbstractKernel{T}
    k::K
    p::Int
end
(P::Power)(τ) = P.k(τ)^P.p
(P::Power)(x, y) = P.k(x, y)^P.p
Base.:^(k::AbstractKernel, p::Int) = Power(k, p)
parameters(k::Power) = parameters(k.k)
nparameters(k::Power) = nparameters(k.k)
function Base.similar(k::Power, θ::AbstractVector)
    Power(similar(k.k, θ), k.p)
end

############################ Separable Product #################################
using LinearAlgebraExtensions: LazyGrid, grid
# product kernel, but separately evaluates component kernels on different parts of the input
struct SeparableProduct{T, K<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel{T}
    args::K # kernel for input covariances
    function SeparableProduct(k::Tuple{Vararg{AbstractKernel}})
        T = promote_type(eltype.(k)...)
        new{T, typeof(k)}(k)
    end
end
parameters(k::SeparableProduct) = vcat(parameters.(k.args)...)
nparameters(k::SeparableProduct) = sum(nparameters, k.args)

function Base.similar(k::SeparableProduct, θ::AbstractVector)
    SeparableProduct(_similar_helper(k, θ))
end

# both x and y have to be vectors of inputs to individual kernels
# could also consist of tuples ... so restricting to AbstractVector might not be good
function (K::SeparableProduct)(x::AbstractVector, y::AbstractVector)
    checklength(x, y)
    val = one(eltype(K))
    for (i, k) in enumerate(K.args)
        val *= k(x[i], y[i])
    end
    return val
end
# if we had kernel input type, could compare with eltype(X)
function gramian(K::SeparableProduct, X::LazyGrid, Y::LazyGrid)
    kronecker((gramian(kxy...) for kxy in zip(K.args, X.args, Y.args))...)
end
function gramian(K::SeparableProduct, X::LazyGrid)
    kronecker((gramian(kx...) for kx in zip(K.args, X.args))...)
end
# TODO: if points are not on a grid, can still evaluate dimensions separately,
# and take elementwise product. Might lead to efficiency gains
############################### Separable Sum ##################################
# what about separable sums? do they give rise to kronecker sums? yes!
struct SeparableSum{T, K<:Tuple{Vararg{AbstractKernel}}} <: AbstractKernel{T}
    args::K # kernel for input covariances
    function SeparableSum(k::Tuple{Vararg{AbstractKernel}})
        T = promote_type(eltype.(k)...)
        new{T, typeof(k)}(k)
    end
end
parameters(k::SeparableSum) = vcat(parameters.(k.args)...)
nparameters(k::SeparableSum) = sum(nparameters, k.args)

function Base.similar(k::SeparableSum, θ::AbstractVector)
    SeparableSum(_similar_helper(k, θ))
end

function (K::SeparableSum)(x::AbstractVector, y::AbstractVector)
    checklength(x, y)
    val = zero(eltype(K))
    for (i, k) in enumerate(K.args)
        val += k(x[i], y[i])
    end
    return val
end

function gramian(K::SeparableSum, X::LazyGrid, Y::LazyGrid)
    ⊕((gramian(kxy...) for kxy in zip(K.args, X.args, Y.args))...)
end

# convenient constructor
# e.g. separable(*, k1, k2)
separable(::typeof(*), k::AbstractKernel...) = SeparableProduct(k)
separable(::typeof(+), k::AbstractKernel...) = SeparableSum(k)
# d-separable product of k
function separable(::typeof(^), k::AbstractKernel, d::Integer)
    SeparableProduct(tuple((k for _ in 1:d)...))
end
