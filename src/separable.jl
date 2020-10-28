# technically, this is a product kernel of a continuous and discrete input kernel ...
struct SeparableKernel{T, K, M<:AbstractMatrix{T}} <: MultiKernel{T}
    k::K # kernel for input covariances
    B::M # matrix for output covariances
    # potentially initialize with different type
end
SeparableKernel(B::AbstractMatrix, k) = SeparableKernel(k, B)
const Separable = SeparableKernel
# outputs matrix-valued covariance
(k::Separable)(x, y) = k.B * k.k(x, y)

Base.size(K::Separable) = size(K.B)
Base.getindex(K::Separable, i::Integer, j::Integer) = K.B[i,j] * K.k

using KroneckerProducts
# this is isotopic marginalization, where we observe all outputs for all inputs
# function gramian(k::Separable, x::AbstractVector, y::AbstractVector = x)
#     k.B ⊗ gramian(k.k, x, y)
# end

function factorize(G::Gramian{<:AbstractMatrix, <:Separable})
    factorize(kronecker(G))
end

function KroneckerProducts.kronecker(G::Gramian{<:AbstractMatrix, <:Separable})
    gramian(G.k.k, G.x, G.y) ⊗ G.k.B
end

# also, need special multiplication
function LinearAlgebra.mul!(y::AbstractVecOfVec, G::Gramian{<:AbstractMatrix, <:Separable},
                            x::AbstractVecOfVec)
    K = G.k.B ⊗ gramian(G.k.k, x, y) # convert to kronecker
    return mul!(y, kronecker(G), x)
end

# ISSUE: kronecker product returns a matrix of scalars, whereas
# gramian of gradient kernel returns matrix of matrices ...
# implement "block" version of kronecker?
