######################### Lazy Difference Vector ###############################
# TODO: is it worth using the LazyArrays package?
# could be replaced by LazyVector(applied(-, x, y)), which is neat.
# lazy difference between two vectors, has no memory footprint
struct LazyDifference{T, U, V} <: AbstractVector{T}
    x::U
    y::V
    function LazyDifference(x, y)
        length(x) == length(y) || throw(DimensionMismatch("x and y do not have the same length: $(length(x)) and $(length(y))."))
        T = promote_type(eltype(x), eltype(y))
        new{T, typeof(x), typeof(y)}(x, y)
    end
end

difference(x::Number, y::Number) = x-y # avoid laziness for scalars
difference(x::AbstractVector, y::AbstractVector) = LazyDifference(x, y)
difference(x::Tuple, y::Tuple) = LazyDifference(x, y)
# this creates a type instability:
# difference(x, y) = length(x) == length(y) == 1 ? x[1]-y[1] : LazyDifference(x, y)
Base.size(d::LazyDifference) = (length(d.x),)
Base.getindex(d::LazyDifference, i::Integer) = d.x[i]-d.y[i]

################################################################################
LinearAlgebra.diagzero(D::Diagonal{<:Diagonal{T}}, i, j) where T = zeros(T, (size(D.diag[i], 1), size(D.diag[j], 2)))

# if we have Matrix valued kernels, this should be different
Base.eltype(k::AbstractKernel{T}) where {T} = T
Base.eltype(::MultiKernel{T}) where {T} = Matrix{T}

# TODO: simplify
const VecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const VecOrVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractVecOfVecOrMat{T} = AbstractVector{<:AbstractVecOrMat{T}}

################################################################################
# euclidean distance
# IDEA: @inline?
function euclidean2(x, y)
    length(x) == length(y) || throw(DimensionMismatch("inputs have to have the same length: $(length(x)), $(length(y))"))
    val = zero(promote_type(eltype(x), eltype(y)))
    @inbounds @simd for i in eachindex(x)
        val += (x[i] - y[i])^2
    end
    return val
end
euclidean(x, y) = sqrt(euclidean2(x, y))

# energetic norm
enorm(A::AbstractMatOrFac, x::AbstractVector) = sqrt(enorm2(A, x))
enorm2(A::AbstractMatOrFac, x::AbstractVector) = dot(x, A, x)

# IDEA: maybe include in and output dimension of kernel in type?
# this makes it easier to type check admissability of input arguments
# and differentiate between data vector or vector of data
# or instead, write run-time input check
# change to a boundscheck?
function checklength(x::AbstractArray, y::AbstractArray)
    lx = length(x)
    ly = length(y)
    if lx != ly; throw(DimensionMismatch("length(x) ($lx) ≠ length(y) ($ly)")) end
    return length(x)
end

# do we need an MVector version of this?
function vector_of_static_vectors(A::AbstractMatrix)
    D = size(A, 1)
    [SVector{D}(col) for col in eachcol(A)]
end
function vector_of_static_vectors(A::AbstractVector{<:Number})
    [SVector{1}(a) for a in A]
end
vector_of_static_vectors(A::Vector{V}) where {V <: Union{SVector, MVector}} = A
# NOTE: element-vectors have
function vector_of_static_vectors(A::Vector{<:AbstractVector})
    # length(A) > 0 || return [SVector{0, eltype(A)}()]
    D = size(A[1], 1)
    [SVector{D}(a) for a in A]
end

# _Matrix(A::AbstractMatrix) = A
# _Matrix(A::Vector{<:Union{SVector, MVector}}) = A
# this first converts a colview back to the corresponding matrix
# function _Matrix(x::Vector{<:SubArray{<:Any, 1, <:Matrix, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}})
#     x[1].parent
# end

################################################################################
# is positive semi-definite
function ispsd end
ispsd(A::Number, tol::Real = 0.) = A ≥ -tol
# TODO: replace with own pivoted cholesky
ispsd(A::AbstractMatrix, tol::Real = 0.) = all(A->ispsd(A, tol), eigvals(A))
iscov(A::AbstractMatrix, tol::Real = 0.) = issymmetric(A) && ispsd(A, tol)
# import LinearAlgebraExtensions: iscov
iscov(A::AbstractMatrix{<:Real}, tol::Real = 0.) = issymmetric(A) && ispsd(A, tol)
iscov(k::MercerKernel, x = randn(32), tol = 1e-10) = iscov(gramian(k, x), tol)

####################### randomized stationarity tests ##########################
# NOTE this test only allows for euclidean isotropy ...
# does not matter for 1d tests
function isisotropic(k::AbstractKernel, x::AbstractVector)
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
# lazy representation of perfect shuffle matrices
# # X is n by m
# S::PerfectShuffle will perform S*vec(X) = vec(X')
struct PerfectShuffle <: AbstractMatrix{Bool}
    n::Int
    m::Int
end

Base.size(S::PerfectShuffle) = (n*m, n*m)
Base.size(S::PerfectShuffle, i::Int) = 0 < i ≤ 2 ? n*m : 1

SparseArrays.SparseMatrixCSC(S::PerfectShuffle) = perfect_shuffle(S.n, S.m)
function LinearAlgebra.mul!(b::AbstractVector, S::PerfectShuffle, a::AbstractVector, α::Number = 1, β::Number = 0)
    length(b) == length(a) == size(S, 1) || throw(DimensionMismatch())
    A, B = reshape(a, S.n, S.m), reshape(b, S.n, S.m)
    if β == 0
        @. B = α * A'
    else
        @. B = α * A' + β * B
    end
    return b
end
Base.:*(S::PerfectShuffle, a::AbstractVector) = mul!(zero(a), S, a)

function perfect_shuffle(n::Int)
    S = spzeros(Bool, n^2, n^2)
    for i in 1:n, j in 1:n
        S[j+n*(i-1), i+n*(j-1)] = 1
    end
    return S
end

# X is n by m
# S will perform S*vec(X) = vec(X')
function perfect_shuffle(n::Int, m::Int)
    S = spzeros(Bool, n*m, n*m)
    for i in 1:n, j in 1:m
        S[j+m*(i-1), i+n*(j-1)] = 1
    end
    return S
end

# exchange matrix (anti-diagonal identity matrix)
function exchange_matrix(r::Int)
    E = spzeros(Bool, r, r)
    for i in 1:r
        E[r-i+1, i] = 1
    end
    return E
end

# function anti_diagonal(x::AbstractVector)
#     i = tuple.(r:-1:1, 1:r)
# end

# computes all products of subsets of length(x)-1 elements of x and stores them in x
# useful for gradient product kernel structure
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
