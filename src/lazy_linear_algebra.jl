# implements lazy matrix products and sums
# in contrast to LazyArrays, it implements a different type of laziness
# LazyArrays is fully lazy and avoids computing temporaries, but assumes the
# entries of the component arrays can be computed easily and efficiently
# The structures below rely on the mul! functions of the constituent
# matrices / factorizations to define a potentially more efficient multiplication\
# IDEA: introduce temporaries and optimal matrix multiplication order to minimize complexity
# IDEA: enable rdiv! for LazyMatrixSum, Product
abstract type LazyFactorization{T} <: Factorization{T} end
# const default_tol = 1e-6

################################################################################
# in contrast to the AppliedMatrix in LazyArrays, this is not completely lazy
# in that it calculates intermediate results, important to make use of special structure
# observation: this is remarkably similar to the KroneckerProducts implementation
# allow AbstractMatOrFacOrUni? will need adjustment for size function
struct LazyMatrixProduct{T, AT, F} <: LazyFactorization{T}
    args::AT
    tol::F
    # temporaries
    # args is vector or tuple of constituent matrices
    function LazyMatrixProduct(args; tol::Real = default_tol)
        for i in 1:length(args)-1
            size(args[i], 2) == size(args[i+1], 1) || throw(DimensionMismatch("$i"))
        end
        T = promote_type(eltype.(args)...)
        new{T, typeof(args), typeof(tol)}(args, tol)
    end
end
function LazyMatrixProduct(A::AbstractMatOrFac...; tol::Real = default_tol)
    args_vec = vcat((lazy_product_flattened_args(Ai) for Ai in A)...)
    args = tuple(args_vec...) # casting to tuple makes it possible to infer types
    LazyMatrixProduct(args, tol = tol)
end
lazy_product_flattened_args(A) = [A]
lazy_product_flattened_args(A::LazyMatrixProduct) = [A.args...]

function Base.size(L::LazyMatrixProduct, i::Int)
    if i == 1
        size(L.args[1], 1)
    elseif i == 2
        size(L.args[end], 2)
    elseif i > 2
        1
    end
end
Base.size(L::LazyMatrixProduct) = size(L, 1), size(L, 2)
Base.adjoint(L::LazyMatrixProduct) = LazyMatrixProduct(reverse(adjoint.(L.args)))
issquare(A::AbstractMatOrFac) = size(A, 1) == size(A, 2)
Base.Matrix(L::LazyMatrixProduct) = prod(Matrix, L.args)
Base.AbstractMatrix(L::LazyMatrixProduct) = prod(AbstractMatrix, L.args)

# function LinearAlgebra.diag(L::LazyMatrixProduct)
#     middle = length(L.args) > 2 ? *(L.args[2:end-1]) : 1
#     n = minimum(size(L))
#     d = zeros(eltype(L), n)
#     for i in 1:n
#         d = @views L.args[1][i, :]' * middle * L.args[end][:, i]
#     end
#     return d
# end
# function LinearAlgebra.factorize(A::LazyMatrixProduct; k::Int = 16, sigma::Real = 1e-2)
#     return CGFact(A, tol = A.tol)
# end

Base.:*(x::AbstractMatrix, L::LazyMatrixProduct) = adjoint(L'*x')
function Base.:*(L::LazyMatrixProduct, x::AbstractVector) # IDEA: could consolidate with LazyMatrixSum
    T = promote_type(eltype(L), eltype(x))
    b = zeros(T, size(L, 1))
    mul!(b, L, x)
end
function Base.:*(L::LazyMatrixProduct, X::AbstractMatrix)
    T = promote_type(eltype(L), eltype(X))
    B = zeros(T, (size(L, 1), size(X, 2)))
    mul!(B, L, X)
end
# IDEA: this could be more memory-efficient if we knew that all matrices had the same size
function LinearAlgebra.mul!(y::AbstractVecOrMat, L::LazyMatrixProduct, x::AbstractVecOrMat, α::Real = 1, β::Real = 0)
    z = deepcopy(x)
    for A in reverse(L.args)
        z = A*z
    end
    @. y = α*z + (iszero(β) ? 0 : β*y)
    return y
end

################################################################################
# in contrast to the AppliedMatrix in LazyArrays, this is not completely lazy
# in that it calculates intermediate results
# TODO: should we have a tol field?
struct LazyMatrixSum{T, AT, F} <: LazyFactorization{T}
    args::AT
    tol::F
    function LazyMatrixSum(args; tol::Real = default_tol)
        all(==(size(args[1])), size.(args)) || throw(DimensionMismatch())
        T = promote_type(eltype.(args)...)
        new{T, typeof(args), typeof(tol)}(args, tol)
    end
end
function LazyMatrixSum(A::AbstractMatOrFac...; tol::Real = default_tol)
    args_vec = vcat((lazy_sum_flattened_args(Ai) for Ai in A)...)
    args = tuple(args_vec...)
    LazyMatrixSum(args, tol = tol)
end
lazy_sum_flattened_args(A) = [A]
lazy_sum_flattened_args(A::LazyMatrixSum) = [A.args...]

Base.size(L::LazyMatrixSum, i...) = size(L.args[1], i...)
Base.adjoint(L::LazyMatrixSum) = LazyMatrixSum(adjoint.(L.args))
Base.Matrix(L::LazyMatrixSum) = sum(Matrix, L.args)
Base.AbstractMatrix(L::LazyMatrixSum) = sum(AbstractMatrix, L.args)
Base.getindex(L::LazyMatrixSum, i...) = sum(x->getindex(x, i...), L.args)
LinearAlgebra.diag(L::LazyMatrixSum) = sum(diag, L.args)

Base.:*(x::AbstractMatrix, L::LazyMatrixSum) = adjoint(L'*x')
function Base.:*(L::LazyMatrixSum, x::AbstractVector)
    T = promote_type(eltype(L), eltype(x))
    b = zeros(T, size(L, 1))
    mul!(b, L, x)
end
function Base.:*(L::LazyMatrixSum, X::AbstractMatrix)
    T = promote_type(eltype(L), eltype(X))
    B = zeros(T, (size(L, 1), size(X, 2)))
    mul!(B, L, X)
end
function LinearAlgebra.mul!(y::AbstractVecOrMat, L::LazyMatrixSum, x::AbstractVecOrMat,
                            α::Real = 1, β::Real = 0)
    @. y = (iszero(β) ? 0 : β*y)
    for A in L.args
        mul!(y, A, x, α, 1)
    end
    return y
end

function LinearAlgebra.:\(A::LazyFactorization, b::AbstractVector)
    T = promote_type(eltype(A), eltype(b))
    x = zeros(T, size(A, 1))
    ldiv!(x, A, b)
end
# solve general BlockGramian via minimum residual solver
function LinearAlgebra.ldiv!(x::AbstractVector, A::LazyFactorization, b::AbstractVector; kwargs...)
    minres!(x, A, b; kwargs...)
end

# IDEA: more efficient temporary allocation
# function LinearAlgebra.mul!(y::AbstractVector, L::LazyMatrixProduct, x::AbstractVector, α::Real = 1, β::Real = 0)
#     if all(issquare, L.args)
#         z1, z2 = similar(x), similar(x)
#
#         # if odd and greater than 1, we need to start with z
#         iseven(length(L.args)) ? copyto!(y, x) : copyto!(z, x)
#         for A in L.args
#             mul!(z, A, y)
#             y, z = z, y
#         end
#
#     else
#         z = copy(x)
#         for A in L.args
#             z = A*z
#         end
#         copyto!(y, z)
#     end
#     return y
# end
