# using LinearAlgebraExtensions: vecofvec
vecofvec(A::AbstractMatrix) = [a for a in eachcol(A)] # one view allocates 64 bytes

# IDEA: use LazyArrays to write efficient gramian for VerticalRescaling and
# InputScaling kernels
############################ Lazy Kernel Matrix ###############################
# note: gramian specializations for special matrix structure has to be after definition of all kernels
# move to MyLazyArrays?
# K can be any kernel, so not necessarily MercerKernel
struct Gramian{T, K, U<:AbstractVector, V<:AbstractVector} <: AbstractMatrix{T}
    k::K # has to allow k(x[i], y[j]) evaluation ∀i,j
    x::U
    y::V
end

const BlockGramian{T, M, K, X, Y} = BlockFactorization{<:T, <:Gramian{M, K, X, Y}}

function Gramian(k, x::AbstractVector, y::AbstractVector = x)
    T = gramian_eltype(k, x[1], y[1])
    Gramian{T, typeof(k), typeof(x), typeof(y)}(k, x, y)
end
# defaults to euclidean dot product
Gramian(x::AbstractVector, y::AbstractVector) = Gramian(Dot(), x, y)

Base.size(K::Gramian) = (length(K.x), length(K.y))
Base.eltype(G::Gramian{T}) where {T} = T
# size of an element of a matrix of matrices
elsize(G::Gramian) = size(G[1, 1])
elsize(G::Gramian{<:Number}) = ()
function gramian_eltype(k::AbstractKernel, x, y)
    promote_type(eltype(k), eltype(eltype(x)), eltype(eltype(y)))
end
gramian_eltype(k, x, y) = typeof(k(x, y)) # default to evaluation

# indexing
# NOTE: @inline helps increase mvm performance by 50%
@inline function Base.getindex(G::Gramian, i::Integer, j::Integer)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds G.k(G.x[i], G.y[j]) # remove boundscheck of x of x and y
end
function Base.getindex(G::Gramian, i, j::Integer)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds @views G.k.(G.x[i], (G.y[j],))
end
function Base.getindex(G::Gramian, i::Integer, j)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds @views G.k.((G.x[i],), G.y[j])
end
function Base.getindex(G::Gramian, i, j)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds @views Matrix(gramian(G.k, G.x[i], G.y[j]))
end

# maintain laziness by default when adding diagonal
function Base.:+(D::Union{Diagonal, BlockDiagonalFactorization}, G::Union{Gramian, BlockGramian})
    LazyMatrixSum(D, G)
end
function Base.:+(G::Union{Gramian, BlockGramian}, D::Union{Diagonal, BlockDiagonalFactorization})
    LazyMatrixSum(G, D)
end

# IDEA: GPU
using LinearAlgebra: checksquare
using Base.Threads
# make generic multiply multi-threaded and SIMD-enabled
function Base.:*(G::Gramian, a::AbstractVector)
    T = promote_type(eltype(G), eltype(a))
    b = zeros(T, size(G, 1))
    mul!(b, G, a)
end
function Base.:*(G::Gramian, A::AbstractMatrix)
    T = promote_type(eltype(G), eltype(A))
    B = zeros(T, size(G, 1), size(A, 2))
    mul!(B, G, A)
end
# NOTE: these specialized multiplication functions scale better for large n
# but tend to be slightly slower than Julia's generic mul! for small n, likely because of parallelization
function LinearAlgebra.mul!(y::AbstractVector, G::Gramian, x::AbstractVector, α::Real = 1, β::Real = 0)
    n, m = size(G)
    @. y = iszero(β) ? 0 : β * y
    @threads for i in 1:n
        @simd for j in 1:m
            @inbounds y[i] += α * G[i, j] * x[j]
        end
    end
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, G::Gramian, X::AbstractMatrix, α::Real = 1, β::Real = 0)
    @. Y = iszero(β) ? 0 : β * Y # makes sure returns correctly if β = 0 and y contains NaNs, e.g. through `similar` initialization
    @threads for j in 1:size(Y, 2)
        for i in 1:size(Y, 1)
            @simd for k in 1:size(X, 1) # indexing order so that dense arrays Y, X are indexed in a Cache-friendly way
                @inbounds Y[i, j] += α * G[i, k] * X[k, j]
            end
        end
    end
    return Y
end

# parallel matrix instantiation
function Base.Matrix(G::Gramian)
    n, m = size(G)
    M = Matrix{eltype(G)}(undef, n, m)
    Matrix!(M, G)
end
function Matrix!(M::AbstractMatrix, G)
    @threads for j in 1:size(G, 2)
        @simd for i in 1:size(G, 1)
            @inbounds M[i, j] = G[i, j]
        end
    end
    return M
end
Base.AbstractMatrix(G::Gramian) = Matrix(G)
Base.adjoint(G::Gramian) = Gramian(G.k, G.y, G.x)
Base.transpose(G::Gramian) = Gramian(G.k, G.y, G.x)

# by default, Gramians of matrix-valued kernels are BlockFactorizations, O(1) memory complexity
function gramian(k::MultiKernel, x::AbstractVector, y::AbstractVector, lazy::Val{true} = Val(true))
    G = Gramian(k, x, y)
    BlockFactorization(G, isstrided = true) # strided because every block has same size
end

# instantiates the blocks but respects structure O(n^2d) memory complexity for gradient kernel
function gramian(k::MultiKernel, x::AbstractVector, y::AbstractVector, lazy::Val{false}) # = Val(false))
    G = Gramian(k, x, y)
    G = Matrix(G)
    BlockFactorization(G, isstrided = true)
end

LinearAlgebra.issymmetric(G::Gramian) = (G.x ≡ G.y) || (G.x == G.y) # pointer check is constant time
LinearAlgebra.issymmetric(G::Gramian{<:Real, <:Constant}) = true
LinearAlgebra.ishermitian(G::Gramian) = issymmetric(G)
function LinearAlgebra.isposdef(G::Gramian)
    return typeof(G.k) <: Union{MercerKernel, MultiKernel} && issymmetric(G)
end

######################### smart pseudo-constructor #############################
# standard approach is a lazily represented kernel matrix
# by default, Gramians of matrix-valued kernels are BlockFactorizations
# TODO: don't return BlockFactorization if one dimension is one!
# elsize(G)[1]  == 1 && size(G)[1] == 1
function gramian(k, x::AbstractVector, y::AbstractVector)
    G = Gramian(k, x, y)
    eltype(G) <: AbstractMatOrFac ? BlockFactorization(G) : G
end
gramian(k, x::AbstractVector) = gramian(k, x, x)

gramian(x::AbstractVector, y::AbstractVector) = Gramian(x, y)
gramian(x::AbstractVector) = gramian(x, x)

# if matrix whose columns are datapoints is passed, convert to vector of vectors
gramian(k, x::AbstractMatrix) = gramian(k, vecofvec(x))
gramian(k, x::AbstractMatrix, y::AbstractMatrix) = gramian(k, vecofvec(x), vecofvec(y))

# if input trait is not used, ignore it
gramian(k, x, y, ::InputTrait) = gramian(k, x, y)
gramian(k, x, ::InputTrait) = gramian(k, x)

# IDEA: gramian(k, x, y, input = input_trait(k))
# makes it possible to take advantage of specialized implementations
# and custom kernels k, by defining input_trait(k)

# 1D stationary kernel on equi-spaced grid yields Toeplitz structure
# works if input_trait(k) <: Union{IsotropicInput, StationaryInput}
gramian(k, x::StepRangeLen, y::StepRangeLen) = gramian(k, x, y, input_trait(k))
gramian(k, x::StepRangeLen, y::StepRangeLen, ::GenericInput) = Gramian(k, x, y)

# IDEA: should this be in factorization? since dft still costs linear amount of information
# while gramian is usually lazy and O(1) in structure construction
function gramian(k, x::StepRangeLen, y::StepRangeLen, ::Union{IsotropicInput, StationaryInput})
    if x === y
        k1 = k.(x[1], x)
        SymmetricToeplitz(k1)
    elseif x.step == y.step
        k1 = k.(x, y[1])
        k2 = k.(x[1], y)
        Toeplitz(k1, k2)
    else
        Gramian(k, x, y)
    end
end

# 1D stationary kernel on equi-spaced grid with periodic boundary conditions
function gramian(k::StationaryKernel, x::StepRangeLen, ::PeriodicInput)
    k1 = k.(x[1], x)
    Circulant(k1)
end

###################### factorization of Gramian ################################
# IDEA: have to use special cholesky implementation to avoid instantiating G in low rank case
function LinearAlgebra.cholesky(G::Gramian, pivoting::Val{true}; check::Bool = false, tol::Real = 0.0)
    cholesky!(Symmetric(Matrix(G)), Val(true), check = check, tol = tol)
end

function LinearAlgebra.cholesky(G::Gramian, pivoting::Val{false} = Val(false); check::Bool = true)
    cholesky!(Symmetric(Matrix(G)), Val(false), check = check)
end

const DEFAULT_MAX_CHOLESKY_SIZE = 2^14
const DEFAULT_TOL = 1e-6

# TODO: add sparsification technique for exponentially decaying kernels in high d
function LinearAlgebra.factorize(G::Gramian; max_cholesky_size::Int = DEFAULT_MAX_CHOLESKY_SIZE,
                                             tol::Real = DEFAULT_TOL)
    n = checksquare(G)
    if n ≤ max_cholesky_size # defaults to pivoted cholesky to detect low rank structure
        cholesky(G, Val(true), check = false, tol = tol)
    else # IDEA: calculate preconditioner here?
        G # if instantiating G is not possible in memory, keep lazy and proceed via cg,
    end
end

# abstract type InferenceType end
# struct CholeskyInference <: InferenceType end
# struct ConjugateGradientInference <: InferenceType end
# struct SparseInference <: InferenceTypeend
#
# function LinearAlgebra.factorize(G::Gramian, ::CholeskyInference)
#     cholesky(G)
# end
#
# function LinearAlgebra.factorize(G::Gramian, ::SparseInference)
#     return -1
# end

###################### Specializations for BlockGramians #######################
function LinearAlgebra.:\(B::BlockGramian, b::AbstractVector)
    T = promote_type(eltype(B), eltype(b))
    x = zeros(T, size(B, 1))
    ldiv!(x, B, b)
end
# solve general BlockGramian via minimum residual solver
# IDEA: cg instead? is faster for high-dimensional (d>64) gradient kernels
function LinearAlgebra.ldiv!(x::AbstractVector, B::BlockGramian, b::AbstractVector; kwargs...)
    cg!(x, B, b; kwargs...)
end

# carries out multiplication for general BlockFactorization
function BlockFactorizations.blockmul!(y::AbstractVecOfVecOrMat, G::Gramian, x::AbstractVecOfVecOrMat, α::Real = 1, β::Real = 0)
    Gijs = [G[1, 1] for _ in 1:Base.Threads.nthreads()] # pre-allocate storage for elements
    IT = input_trait(G.k)
    @threads for i in eachindex(y) # NOTE: without threading, this does not allocate anything
        @. y[i] = iszero(β) ? 0 : β * y[i] # makes sure returns correctly if β = 0 and y contains NaNs, e.g. through `similar` initialization
        Gij = Gijs[Base.Threads.threadid()]
        for j in eachindex(x)
            Gij = evaluate_block!(Gij, G, i, j, IT) # this is change to original blockmul!, allowing evaluation of Gramian's elements without additional allocations
            mul!(y[i], Gij, x[j], α, 1)
        end
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVecOfVecOrMat, G::Gramian, x::AbstractVecOfVecOrMat, α::Real = 1, β::Real = 0)
    BlockFactorizations.blockmul!(y, G, x, α, β)
end

# IDEA: @inline?
function evaluate_block!(Gij, G::Gramian, i::Int, j::Int, T = input_trait(G.k))
    evaluate_block!(Gij, G.k, G.x[i], G.y[j], T)
end

# if target is number, just evaluate the gramian
function evaluate_block!(Gij::Number, G::Gramian, i::Int, j::Int, T = input_trait(G.k))
    G[i, j]
end

# fallback
function evaluate_block!(Gij, k, x, y, T = input_trait(k))
    k(x, y)
end
