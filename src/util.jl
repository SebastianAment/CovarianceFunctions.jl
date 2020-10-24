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

################################################################################
# matrix of matrices to matrix, useful for testing Gramians of multi-output kernels
function matmat2mat(A::AbstractMatrix{<:AbstractMatrix})
    n, m = size(A)
    all(==(size(A[1])), (size(Ai) for Ai in A)) || throw(DimensionMismatch("component matrices do not have the same size"))
    ni, mi = size(A[1])
    B = zeros(n*ni, m*mi)
    for j in 1:m
        jnd = (j-1)*mi+1 : j*mi
        for i in 1:n
            ind = (i-1)*ni+1 : i*ni
            Bij = @view B[ind, jnd]
            copyto!(Bij, A[i, j])
        end
    end
    return B
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
function isstationary(k::MercerKernel, x::AbstractVector)
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
