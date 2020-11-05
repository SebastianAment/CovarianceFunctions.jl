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
function (K::SeparableProduct)(x, y)
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

# TODO: check input lengths
function (K::SeparableSum)(x, y)
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

######################## Kernel Input Transfomations ###########################
############################ Symmetric Kernel ##################################
# make this useable for multi-dimensional inputs!
# in more dimensions, could have more general axis of symmetry
struct SymmetricKernel{T, K<:AbstractKernel} <: AbstractKernel{T}
    k::K # kernel to be symmetrized
    z::T # center
end
parameters(k::SymmetricKernel) = vcat(parameters(k.k), k.z)
nparameters(k::SymmetricKernel) = nparameters(k.k) + 1
function Base.similar(k::SymmetricKernel, θ::AbstractVector)
    n = checklength(k, θ)
    k = similar(k.k, @view(θ[1:n-1]))
    SymmetricKernel(k, θ[n])
end
# const Sym = Symmetric
SymmetricKernel(k::AbstractKernel{T}) where T = SymmetricKernel(k, zero(T))

# for 1D axis symmetry
function (k::SymmetricKernel)(x, y)
    x -= k.z; y -= k.z;
    k.k(x, y) + k.k(-x, y)
end

######################## Kernel output transformations #########################
############################## rescaled kernel #################################
# TODO: should be called DiagonalRescaling in analogy to the matrix case
# diagonal rescaling of covariance functions
# generalizes multiplying by constant kernel to multiplying by function
struct VerticalRescaling{T, K<:AbstractKernel{T}, F} <: AbstractKernel{T}
    k::K
    a::F
end
parameters(k::VerticalRescaling) = vcat(parameters(k.k), parameters(k.a))
nparameters(k::VerticalRescaling) = nparameters(k.k) + nparameters(k.a)

(k::VerticalRescaling)(x, y) = k.a(x) * k.k(x, y) * k.a(y)
# TODO: preserve structure if k.k is stationary and x, y are regular grids,
# since that introduces Toeplitz structure
# function gramian(k::VerticalRescaling, x::AbstractVector, y::AbstractVector)
#     # Diagonal(k.a.(x)) * gramian(k.k, x, y) * Diagonal(k.a.(y)))
#     LazyProduct(Diagonal(k.a.(x)), gramian(k.k, x, y), Diagonal(k.a.(y))))
# end

# normalizes an arbitary kernel so that k(x,x) = 1
normalize(k::AbstractKernel) = VerticalRescaling(k, x->1/√k(x, x))
