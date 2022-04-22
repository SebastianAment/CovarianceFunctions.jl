# technically, this is a product kernel of a continuous and discrete input kernel ...
struct SeparableKernel{T, K, M<:AbstractMatOrFac{T}} <: MultiKernel{T}
    k::K # kernel for input covariances
    B::M # matrix for output covariances
    # potentially initialize with different type
end
function SeparableKernel(B::AbstractMatrix, k)
    T, K, M = promote_type(eltype(B), eltype(k)), typeof(k), typeof(B)
    SeparableKernel{T, K, M}(k, B)
end
const Separable = SeparableKernel
# outputs matrix-valued covariance
(k::Separable)(x, y) = k.B * k.k(x, y)

Base.size(K::Separable) = size(K.B)
Base.getindex(K::Separable, i::Integer, j::Integer) = K.B[i,j] * K.k

# NOTE: maybe this is not worth it for small d!
function evaluate_block!(Gij::AbstractMatrix, k::SeparableKernel, x, y, T = input_trait(k))
    kxy = k.k(x, y)
    @. Gij = B * kxy
end

# this is isotopic marginalization, where we observe all outputs for all inputs
# function gramian(k::Separable, x::AbstractVector, y::AbstractVector = x)
#     k.B ⊗ gramian(k.k, x, y)
# end

function factorize(G::Gramian{<:AbstractMatrix, <:Separable})
    factorize(kronecker(G))
end

function KroneckerProducts.kronecker(G::Gramian{<:Any, <:Separable})
    gramian(G.k.k, G.x, G.y) ⊗ G.k.B
end

# also, need special multiplication
function LinearAlgebra.mul!(y::AbstractVecOfVec, G::Gramian{<:AbstractMatrix, <:Separable},
                            x::AbstractVecOfVec)
    K = gramian(G.k.k, G.x, G.y) ⊗ G.k.B # convert to kronecker # order?
    return mul!(y, kronecker(G), x)
end

# simply applies kernel in d dimensions, useful to implement T-SNE acceleration via matrix-valued Barnes-Hut
# struct DiagonalKernel{T, KT} <: MultiKernel{T}
#     k::KT
#     d::Int
# end
#
# (D::DiagonalKernel)(x, y) = (D.k(x, y)*I)(d)
#
# function evaluate_block!(Gij::Diagonal, k::DiagonalKernel, x, y, T = input_trait(k))
#     Gij.diag .= k.k(x, y)
# end
