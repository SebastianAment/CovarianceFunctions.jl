using ToeplitzMatrices: Circulant, SymmetricToeplitz
using SparseArrays
using LinearAlgebra
include("Kernel.jl")
using .Kernel


struct EmbededToeplitz{T,S} <: AbstractMatrix{T}
    C::Circulant{T,S}
end

# Represents (W)(Ku)(W^T) + D
struct StructuredKernelInterpolant{T,S} <: Factorization{T}
    Ku::EmbededToeplitz{T,S}
    W::SparseMatrixCSC{T,Int}
    d::Vector{T}
end

# Constructs the Circulant matrix out of a kernel and a given range of points to
# grid
function ip_grid(k, grid)
    G = gramian(k, grid)
    v = G[1, :]
    v′ = [v:v[end - 1:-1:2]]
    return Circulant(v')
end

# k is kernel, x is a vector of data, and m is the number of grid points
function structured_kernel_interpolant(k, x, m)
    Ku = ip_grid(k, range(min(x), max(x), length = m))
    W = Matrix(I, size(Ku)...)
    d = diag(gramian(k, x))
    for i in 1:length(x)
        d[i] -= W[:, i]' * (Ku * (W[:, i]))
    end
    return StructuredKernelInterpolant(Ku, W, d)
end

# Gives dimension for the given n x n matrix. Outputs a tuple of the dimensions
# of the matrix. 
size(S::StructuredKernelInterpolant) = length(S.d), length(S.d)

# Left division. Equivalent to (S^-1)b. Overwrites b. Returns nothing
function ldiv!(S::StructuredKernelInterpolant, b)
    # TODO
    return nothing
end

# Takes the determinent of S. Returns a scalar
det(S::StructuredKernelInterpolant) = exp(logdet(S))

# Equivalent to ln(det(S)). Returns a scalar
function logdet(S::StructuredKernelInterpolant)
    # TODO
    return 1
end

# Returns the inverse of the matrix S. 
inv(S::StructuredKernelInterpolant) = ldiv!(S, Matrix(I, size(S)...))

# Returns true if S is positive definite and false otherwise. 
function isposdef(S::StructuredKernelInterpolant)
    return det(S) > 0
end