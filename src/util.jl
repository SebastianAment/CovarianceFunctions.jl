################################################################################
LinearAlgebra.diagzero(D::Diagonal{<:Diagonal{T}}, i, j) where T = zeros(T, (size(D.diag[i], 1), size(D.diag[j], 2)))

# if we have Matrix valued kernels, this should be different
Base.eltype(k::AbstractKernel{T}) where {T} = T
Base.eltype(::MultiKernel{T}) where {T} = Matrix{T}

const VecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const VecOrVecOfVec{T} = AbstractVector{<:AbstractVector{T}}

################################################################################
# euclidean distance
_d2(x::Real, y::Real) = (x-y)^2
_d2(x::Tuple) = _d2(x...)
euclidean2(x, y) = sum(_d2, zip(x, y))
euclidean(x, y) = sqrt(_euclidean2(x, y))

# energetic norm
enorm(A::AbstractMatOrFac, x::AbstractVector) = sqrt(dot(x, A, x))

import LinearAlgebraExtensions: iscov
iscov(k::MercerKernel, x = randn(32), tol = 1e-10) = iscov(gramian(k, x), tol)

# TODO: maybe include in and output dimension of kernel in type?
# this makes it easier to type check admissability of input arguments
# and differentiate between data vector or vector of data
# TODO: or instead, write run-time input check
# change to a boundscheck?
function checklength(x::AbstractArray, y::AbstractArray)
    lx = length(x)
    ly = length(y)
    if lx != ly; throw(DimensionMismatch("length(x) ($lx) ≠ length(y) ($ly)")) end
    return length(x)
end

####################### randomized stationarity tests ##########################
# WARNING this test only allows for euclidean isotropy ...
# does not matter for 1d tests
function isisotropic(k::MercerKernel, x::AbstractVector)
    isiso = false
    if isstationary(k)
        isiso = true
        r = norm(x[1]-x[2])
        println(r)
        kxr = k(x_i/r, x_j/r)
        for x_i in x, x_j in x
            r = norm(x_i-x_j)
            val = k(x_i/r, x_j/r)
            if !(kxr ≈ val)
                isiso = false
            end
        end
    end
    return isiso
end

# tests if k is stationary on finite set of points x
# need to make this multidimensional
function isstationary(k::AbstractKernel, x::AbstractVector)
    n = length(x)
    d = length(x[1])
    is_stationary = true
    for i in 1:n, j in 1:n
        ε = eltype(x) <: AbstractArray ? randn(d) : randn()
        iseq = k(x[i], x[j]) ≈ k(x[i]+ε, x[j]+ε)
        if !iseq
            println(i ,j)
            println(x[i], x[j])
            println(k(x[i], x[j]) - k(x[i]+ε, x[j]+ε))
            is_stationary = false
            break
        end
    end
    K = typeof(k)
    if !is_stationary && isstationary(k)
        println("Covariance function " * string(K) * " is non-stationary but kernel k returns isstationary(k) = true.")
        return false
    end

    # if the kernel seems stationary but isn't a subtype of stationary kernel, suggest making it one
    if is_stationary && !isstationary(k)
        println("Covariance function " * string(K) * " seems to be stationary. Consider defining isstationary(k::$K) = true.")
    end
    return is_stationary
end

######################### perfect shuffle matrices #############################
function perfect_shuffle(n::Int)
    S = spzeros(n^2, n^2)
    for i in 1:n, j in 1:n
        S[j+n*(i-1), i+n*(j-1)] = 1
    end
    return S
end

# X is n by m
# S will perform S*vec(X) = vec(X')
function perfect_shuffle(n::Int, m::Int)
    S = spzeros(n*m, n*m)
    for i in 1:n, j in 1:m
        S[j+m*(i-1), i+n*(j-1)] = 1
    end
    return S
end

# exchange matrix (anti-diagonal identity matrix)
function exchange_matrix(r::Int)
    E = spzeros(r, r)
    for i in 1:r
        E[r-i+1, i] = 1
    end
    return E
end

# function anti_diagonal(x::AbstractVector)
#     i = tuple.(r:-1:1, 1:r)
# end

# computes all products of subsets of length(x)-1 elements of x and stores them in x
function leave_one_out_products!(x::AbstractArray)
    nz = sum(iszero, x)
    if nz == 0
        @. x = $prod(x) / x
    else
        @. x = 0 # with more than two zeros, all leave-one-out products will be zero
        if nz == 1
            i = findfirst(iszero, x)
            x[i] = prod(x[1:length(x) .!= i])
        end
    end
    return x
end
