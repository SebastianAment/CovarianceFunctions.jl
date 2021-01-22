using ToeplitzMatrices: Circulant, SymmetricToeplitz
using LinearAlgebraExtensions: vecofvec

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
function Gramian(k::AbstractKernel, x::AbstractVector, y::AbstractVector)
    T = promote_type(fieldtype.((k, x, y))...)
    if eltype(k) <: AbstractMatrix
        T = Matrix{T}
    end
    Gramian{T, typeof(k), typeof(x), typeof(y)}(k, x, y)
end
function Gramian(k, x::AbstractVector, y::AbstractVector)
    T = typeof(k(x[1], y[1])) # if we can't directly infer output type, evaluate
    Gramian{T, typeof(k), typeof(x), typeof(y)}(k, x, y)
end
# with euclidean dot product
Gramian(x::AbstractVector, y::AbstractVector) = Gramian(Dot(), x, y)

Base.size(K::Gramian) = (length(K.x), length(K.y))
Base.eltype(G::Gramian{T}) where {T} = T
# size of an element of a matrix of matrices
elsize(G::Gramian{<:AbstractMatrix}) = size(G[1,1])

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

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
# recursively calls mul!, thereby avoiding memory allocation of
# block-matrix multiplication
# IDEA: GPU
function LinearAlgebra.mul!(x::AbstractVecOfVec, G::Gramian{<:Matrix}, y::AbstractVecOfVec)
# function recmul!(x::AbstractVecOfVec, G::Gramian{<:Matrix}, y::AbstractVecOfVec)
    d = elsize(G, 1)
    all(==(d), (length(xi) for xi in x)) || throw(DimensionMismatch("component vectors do not have the same size as component matrices"))
    for j in eachindex(y)
        for i in eachindex(x)
            mul!(x[i], G[i, j], y[j])
        end
    end
    return x
end

LinearAlgebra.issymmetric(G::Gramian) = (G.x ≡ G.y) || (G.x == G.y) # pointer check is constant time
LinearAlgebra.issymmetric(G::Gramian{<:Real, <:Constant}) = true
LinearAlgebra.ishermitian(G::Gramian) = issymmetric(G)
function LinearAlgebra.isposdef(G::Gramian)
    return typeof(G.k) <: Union{MercerKernel, MultiKernel} && issymmetric(G)
end

######################### smart pseudo-constructor #############################
# standard approach is a lazily represented kernel matrix
gramian(k, x::AbstractVector, y::AbstractVector) = Gramian(k, x, y)
gramian(k, x) = gramian(k, x, x)

gramian(x::AbstractVector, y::AbstractVector) = Gramian(x, y)
gramian(x::AbstractVector) = gramian(x, x)

# if matrix whose columns are datapoints is passed, convert to vector of vectors
gramian(k, x::AbstractMatrix) = gramian(k, vecofvec(x))
function gramian(k, x::AbstractMatrix, y::AbstractMatrix)
    gramian(k, vecofvec(x), vecofvec(y))
end

# 1D stationary kernel on equi-spaced grid with periodic boundary conditions
function gramian(k::StationaryKernel, x::StepRangeLen{<:Real}, ::Val{true})
    return Circulant(k.(x[1], x))
end

###################### factorization of Gramian ################################
# IDEA: + hierarchical matrices?
# factorization of Gramian defaults to pivoted Cholesky factorization to exploit low-rank structure
function LinearAlgebra.cholesky(G::Gramian, ::Val{true} = Val(true);
                                check::Bool = true, tol::Real = 1e-12)
    if issymmetric(G) # pivoted cholesky
        return cholesky(Symmetric(G), Val(true); check = check, tol = tol)
    else
        throw("factorize for non-symmetric kernel matrix not implemented")
    end
end
# regular (non-pivoted) cholesky
function LinearAlgebra.cholesky(G::Gramian, ::Val{false}; check::Bool = true)
    if issymmetric(G) # cholesky
        return cholesky(Symmetric(G), Val(false); check = check)
    else
        throw("factorize for non-symmetric kernel matrix not implemented")
    end
end
# IDEA: could decide whether or not to apply pivoted cholesky depending on
# kernel type and input dimension
function LinearAlgebra.factorize(G::Gramian; check::Bool = true, tol::Real = 1e-12)
    return cholesky(G, Val(true), check = check, tol = tol)
end
