# implementation of efficient direct methods for symmetric Toeplitz matrices
using LinearAlgebra
using LinearAlgebra: checksquare
using ToeplitzMatrices

# using ScalableGaussianProcesses
# using FastGaussianProcesses

# computes y = K \ (-r) where K = SymmetricToeplitz([1, r[1:end-1]])
# K is assumed to be positive definite
# r is of length n
durbin(r::AbstractVector) = durbin!(zero(r), r)
# z is a temporary that's necessary for the computation
function durbin!(y::AbstractVector, r::AbstractVector)
    n = length(r)
    length(y) == n || throw(DimensionMismatch("length(y) = $(length(y)) ≠ $n = length(r)"))
    y[1] = -r[1]
    α, β = -r[1], 1
    for k in 1:n-1
        β *= (1-α^2)
        r_k, y_k = @views r[1:k], y[1:k]
        α = @views -(r[k+1] + reverse_dot(r_k, y_k)) / β
        reverse_increment!(y_k, y_k, α)
        y[k+1] = α
    end
    return y
end

# Trench's algorithm (see page 213 of Golub & van Loan)
# computes inverse of symmetric positive definite Toeplitz matrix
function trench(T::SymmetricToeplitz)
    n = checksquare(T)
    B = zeros(eltype(T), n, n)
    trench!(B, T)
end

function trench!(B::AbstractMatrix, T::SymmetricToeplitz)
    r_0 = T.vc[1]
    r = @view T.vc[2:end]
    if r_0 == 1 # levinson implementation assumes normalization of diagonal
        r /= r_0
    end
    trench!(B, r)
    B *= r_0
end

# computes inverse of K = SymmetricToeplitz(vcat(1, r))
function trench(r::AbstractVector)
    n = length(r) + 1
    B = zeros(eltype(r), n, n)
    trench!(B, r)
end

# computes inverse of K = SymmetricToeplitz(vcat(1, r))
# uses B to store the inverse
# NOTE: only populates entries on upper triangle since the inverse is symmetric
function trench!(B::AbstractMatrix, r::AbstractVector)
    n = checksquare(B)
    n == length(r) + 1 || throw(DimensionMismatch())
    y = durbin(r)
    γ = inv(1 + dot(r, y))
    ν = γ * y[end:-1:1]
    B[1, 1] = γ
    @. B[1, 2:n] = γ * y
    for j in 2:n
        for i in 2:j
            @inbounds B[i, j] = B[i-1, j-1] + (ν[n+1-j] * ν[n+1-i] - ν[i-1] * ν[j-1]) / γ
        end
    end
    return Symmetric(B)
end

# solves K \ b where K_{ij} = r[abs(i-j)] and by assumption r[0] = 1
# i.e. K = SymmetricToeplitz(vcat(1, r))
# also, K is assumed to be positive definite
levinson(r::AbstractVector, b::AbstractVector) = levinson!(zero(b), r, b)
function levinson!(x::AbstractVector, r::AbstractVector, b::AbstractVector,
                   y::AbstractVector = zero(x))
    n = length(r) + 1
    length(x) == n || throw(DimensionMismatch("length(x) = $(length(x)) ≠ $n = length(r) + 1"))
    length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)) ≠ $n = length(r) + 1"))
    y[1] = -r[1]
    x[1] = b[1]
    α, β = -r[1], 1
    @inbounds for k in 1:n-1
        β *= (1-α^2)
        r_k, x_k, y_k  = @views r[1:k], x[1:k], y[1:k]
        μ = @views (b[k+1] - reverse_dot(r_k, x_k)) / β
        reverse_increment!(x_k, y_k, μ)
        x[k+1] = μ
        if k < n-1
            α = -(r[k+1] + reverse_dot(r_k, y_k)) / β
            reverse_increment!(y_k, y_k, α) # computes y_k += α * reverse(y_k)
            y[k+1] = α
        end
    end
    return x
end

function levinson(T::SymmetricToeplitz, b::AbstractVector)
    r_0 = T.vc[1]
    r = @view T.vc[2:end]
    if r_0 == 1 # levinson implementation assumes normalization of diagonal
        r /= r_0
    end
    x = levinson(r, b)
    if r_0 != 1
        x ./= r_0
    end
    return x
end

# computes dot(x, reverse(y)) efficiently
function reverse_dot(x::AbstractArray, y::AbstractArray)
    n = length(x)
    n == length(y) || throw(DimensionMismatch())
    d = zero(promote_type(eltype(x), eltype(y)))
    @inbounds @simd for i in 1:n
        d += x[i] * y[n-i+1]
    end
    return d
end
# computes x += α * reverse(y) efficiently without temporary
# NOTE: works without allocations even when x === y
function reverse_increment!(x::AbstractArray, y::AbstractArray = x, α::Number = 1)
    n = length(x)
    n == length(y) || throw(DimensionMismatch())
    if x === y
        @inbounds @simd for i in 1:n÷2
            y_i = y[i] # important to store these for the iteration so that they aren't mutated
            z_i = y[n-i+1]
            x[i] += α * z_i
            x[n-i+1] += α * y_i
        end
        if isodd(n)
            i = (n÷2) + 1 # midpoint
            x[i] += α * y[i]
        end
    else
        @inbounds @simd for i in 1:n
            x[i] += α * y[n-i+1]
        end
    end
    return x
end

# via Bareiss algorithm
# function LinearAlgebra.det(T::SymmetricToeplitz)
#     cholesky()
# end
