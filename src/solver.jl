using ToeplitzMatrices: Circulant, SymmetricToeplitz
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Kernel


struct EmbeddedToeplitz{T,S} <: AbstractMatrix{T}
    C::Circulant{T,S}
end

EmbeddedToeplitz(v) = @views EmbeddedToeplitz(Circulant([v; v[end - 1:-1:2]]))

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

#####
##### Construction - based on Local Quntic Interpolation
##### Fast SKI diagonal construction
##### and Eric Hans Lee's MATLAB code GP_Derivatives
#####

function interp_grid(train_pts, grid_pts)
    n, d = size(train_pts)
    N = length(grid_pts)
    sel_pts, wt = _select_gridpoints(train_pts, grid_pts)

end

function _select_gridpoints(train_vector, grid) 
    stepsize = grid[2] - grid[1]
    J = floor.(Int, (train_vector .- grid[1]) ./ stepsize)
    idx = collect.(J .- 2:J .+ 3) # TODO - Ask Eric if this should be J-1:J+4
    return idx, @views _lq_interp.((train_vector .- grid[idx]) ./ stepsize)
end

# Local Quintic Interpolation
# Key's Cubic Convolution Interpolation Function
function _lq_interp(x)
    x′ = abs(x)
    q = if x′ <= 1
        ((( -0.84375 * x′ + 1.96875) * x′^2) - 2.125) .* x.^2 + 1
    elseif x′ <= 2
        term1 = (0.203125 * x′ - 1.3125) * x′ + 2.65625
        ((term1 * x′ - 0.875) * x′ - 2.578125) * x′ + 1.90625
    elseif x′ <= 3
        term2 = (0.046875 * x′ - 0.65625) * x′ + 3.65625
        ((term2 * x′ - 10.125) * x′ + 13.921875) * x′ - 7.59375
    else
        0
    end
    return q
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
    W = SparseMatrixCSC{Float64,Int}(I, size(Ku)...)
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


function lanczos_arpack(A, k, v; maxiter, tol)
    T = eltype(A)
    n = size(A, 1)
    mulA! = (y, x)->mul!(y, A, x) 
    id = x->x
    # in: (T, mulA!, mulB, solveSI, n, issym, iscmplx, bmat,
    #            nev, ncv, whichstr, tol, maxiter, mode, v0)
    # out: (resid, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, TOL)
    out = Arpack.aupd_wrapper(T, mulA!, id, id, n, true, false, "I",
                       1, k, "LM", tol, maxiter, 1, v)

    α = out[7][k + 1:2 * k - 1]
    β = out[7][2:k - 1]
    
    return out[2], α, β, out[1]
end

function _lanczos_logdet!(z, acc, A, k; maxiter, tol, nsamples)
    for i in 1:nsamples
        rand!(Normal(), z)
        z .= sign.(z)
        Q, α, β, resid = lanczos_arpack(A, k, z; maxiter = maxiter, tol = tol)
        T = SymTridiagonal(α, β)
        Λ = eigen(T)
        wts = Λ.vectors[1, :].^2 .* norm(z)^2
        acc += dot(wts, log.(Λ.values))
    end
    return acc / nsamples
end

# Returns the inverse of the matrix S. 
inv(S::StructuredKernelInterpolant) = ldiv!(S, Matrix(I, size(S)...))

# Returns true if S is positive definite and false otherwise. 
function isposdef(S::StructuredKernelInterpolant)
    return det(S) > 0
end
