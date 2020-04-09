using ToeplitzMatrices: Circulant, SymmetricToeplitz
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Kernel


struct EmbeddedToeplitz{T,S} <: AbstractMatrix{T}
    C::Circulant{T,S}
end

EmbeddedToeplitz(v) = EmbeddedToeplitz(Circulant([v; v[end-1:-1:2]]))

# Computes ABα + Cβ, writing over C
function mul!(c, A::EmbeddedToeplitz, b)
    z = zeros(size(A.C, 1))
    z2 = zeros(size(A.C, 1))
    @views z[1:size(A, 1), :] .= b
    mul!(z2, A.C, z)
    @views c .= z2[1:size(A, 1)]
    return c
end

function size(A::EmbeddedToeplitz) 
    n = (size(A.C, 1) + 2) ÷ 2
    return n, n
end

getindex(A::EmbeddedToeplitz, args...) = getindex(A.C, args...)

# Represents (W)(Ku)(W^T) + D
struct StructuredKernelInterpolant{T,S} <: Factorization{T}
    Ku::EmbeddedToeplitz{T,S}
    W::SparseMatrixCSC{T,Int}
    d::Vector{T}
end

function mul!(c, S::StructuredKernelInterpolant, x)
    c .= S.W * (S.Ku * (S.W' * x)) .+ S.d .* x
    return nothing
end

# k is kernel, x is a vector of data, and m is the number of grid points
function structured_kernel_interpolant(k, x, m)
    G = Kernel.gramian(k, range(minimum(x), maximum(x), length = m))
    v = G[1, :]
    Ku = EmbeddedToeplitz(v)
    W = SparseMatrixCSC{Float64, Int}(I, size(Ku)...)
    d = diag(Kernel.gramian(k, x))
    for i in 1:length(x)
        d[i] -= W[:, i]' * (Ku * (W[:, i]))
    end
    return StructuredKernelInterpolant(Ku, W, d)
end

# Gives dimension for the given n x n matrix. Outputs a tuple of the dimensions
# of the matrix. 
size(S::StructuredKernelInterpolant) = length(S.d), length(S.d)
size(S::StructuredKernelInterpolant, d) = length(S.d) # TODO bounds check

# Left division. Equivalent to (S^-1)b. Overwrites b. Returns nothing
function ldiv!(S::StructuredKernelInterpolant, b)
    x = similar(b)
    cg!(x, S, b)
    b .= x
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
