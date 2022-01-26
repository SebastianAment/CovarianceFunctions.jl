using ToeplitzMatrices: Circulant, SymmetricToeplitz
# using LinearAlgebraExtensions: vecofvec
vecofvec(A::AbstractMatrix) = [a for a in eachcol(A)] # one view allocates 64 bytes

# IDEA: use LazyArrays to write efficient gramian for VerticalRescaling and
# InputScaling kernels
############################ Lazy Kernel Matrix ###############################
# note: gramian specializations for special matrix structure has to be after definition of all kernels
# move to MyLazyArrays?
# K can be any kernel, so not necessarily MercerKernel
struct Gramian{T, K, U<:AbstractVector,
                V<:AbstractVector} <: AbstractMatrix{T}
    k::K # has to allow k(x[i], y[j]) evaluation ∀i,j
    x::U
    y::V
end

const BlockGramian = BlockFactorization{<:Number, <:Gramian}

function Gramian(k, x::AbstractVector, y::AbstractVector = x)
    T = typeof(k(x[1], y[1])) # if we can't directly infer output type, evaluate
    Gramian{T, typeof(k), typeof(x), typeof(y)}(k, x, y)
end
# with euclidean dot product
Gramian(x::AbstractVector, y::AbstractVector) = Gramian(Dot(), x, y)

Base.size(K::Gramian) = (length(K.x), length(K.y))
Base.eltype(G::Gramian{T}) where {T} = T
# size of an element of a matrix of matrices
elsize(G::Gramian) = size(G[1, 1])

# indexing
function Base.getindex(G::Gramian, i::Integer, j::Integer)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds G.k(G.x[i], G.y[j]) # remove boundscheck of x of x and y
end

# TODO: should we make this a view?
function Base.getindex(G::Gramian, i::Union{AbstractArray, Colon},
                                                j::Union{AbstractArray, Colon})
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds gramian(G.k, G.x[i], G.y[j])
end

# IDEA: GPU
using LinearAlgebra: checksquare
using Base.Threads
# probably not necessary anymore, with BlockFactorization
function Base.:*(G::Gramian{<:AbstractMatOrFac{<:Real}}, x::AbstractVecOfVec{<:Number})
    mul!(deepcopy(x), G, x)
end
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
function LinearAlgebra.mul!(y::AbstractVector, G::Gramian, x::AbstractVector, α::Real = 1, β::Real = 0)
    n, m = size(G)
    y .*= β
    @threads for i in 1:n
        @simd for j in 1:m
            @inbounds y[i] += α * G[i, j] * x[j]
        end
    end
    return y
end
# parallel matrix instantiation
function Base.Matrix(G::Gramian)
    n, m = size(G)
    M = Matrix{eltype(G)}(undef, n, m)
    @threads for i in 1:n
        for j in 1:m
            M[i, j] = G[i, j]
        end
    end
    return M
end
Base.AbstractMatrix(G::Gramian) = Matrix(G)
Base.adjoint(G::Gramian) = Gramian(G.k, G.y, G.x)
Base.transpose(G::Gramian) = Gramian(G.k, G.y, G.x)

# need this for blockmul! to work in BlockFactorization
# specialization for Gramians of matrix-valued kernels
# IDEA: precompute input_trait?
function BlockFactorizations.evaluate_block!(Gij, G::Gramian{<:Any, <:MultiKernel}, i::Int, j::Int)
    BlockFactorizations.evaluate_block!(Gij, G.k, G.x[i], G.y[j], input_trait(G.k))
end

# by default, Gramians of matrix-valued kernels are BlockFactorizations, O(1) memory complexity
function gramian(k::MultiKernel, x::AbstractVector, y::AbstractVector, lazy::Val{true} = Val(true))
    G = Gramian(k, x, y)
    BlockFactorization(G)
end

# instantiates the blocks but respects structure O(n^2d) memory complexity for gradient kernel
function gramian(k::MultiKernel, x::AbstractVector, y::AbstractVector, lazy::Val{false})
    G = Gramian(k, x, y)
    G = Matrix(G)
    BlockFactorization(G)
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

# IDEA: gramian(k, x, y, input = input_trait(k))
# makes it possible to take advantage of specialized implementations
# and custom kernels k, by defining input_trait(k)

# 1D stationary kernel on equi-spaced grid yields Toeplitz structure
# IDEA: should this be in factorization? since dft still costs linear amount of information
# while gramian is usually lazy and O(1) in structure construction
function gramian(k::StationaryKernel, x::StepRangeLen{<:Real})
    k1 = k.(x[1], x)
    SymmetricToeplitz(k1)
end

# 1D stationary kernel on equi-spaced grid with periodic boundary conditions
function gramian(k::StationaryKernel, x::StepRangeLen{<:Real}, ::PeriodicInput)
    k1 = k.(x[1], x)
    Circulant(k1)
end

###################### factorization of Gramian ################################
# IDEA: have to use special cholesky implementation to avoid instantiating G in low rank case
function LinearAlgebra.cholesky(G::Gramian, pivoting::Val{true}; check::Bool = false, tol::Real = 0.0)
    cholesky!(Matrix(G), Val(true), check = check, tol = tol)
end

function LinearAlgebra.cholesky(G::Gramian, pivoting::Val{false} = Val(false); check::Bool = true)
    cholesky!(Matrix(G), Val(false), check = check)
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

function LinearAlgebra.:\(B::BlockGramian, b::AbstractVector)
    T = promote_type(eltype(B), eltype(b))
    x = zeros(T, size(B, 1))
    ldiv!(x, B, b)
end
function LinearAlgebra.ldiv!(x::AbstractVector, B::BlockGramian, b::AbstractVector)
    cg!(x, B, b)
end
