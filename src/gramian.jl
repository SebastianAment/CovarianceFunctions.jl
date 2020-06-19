using ToeplitzMatrices: Circulant, SymmetricToeplitz
using LinearAlgebraExtensions: vecofvec

############################ Lazy Gramian Matrix ###############################
# TODO: move to MyLazyArrays?
# Matrix of inner products,
# K can be generalized to be any inner product, so not necessarily MercerKernel
struct Gramian{T, K, U<:AbstractVector,
                V<:AbstractVector} <: AbstractMatrix{T}
    k::K # has to allow k(x[i], y[j]) evaluation ∀i,j
    x::U
    y::V
    function Gramian(k, x::AbstractVector, y::AbstractVector)
        T = promote_type(fieldtype.((k, x, y))...)
        new{T, typeof(k), typeof(x), typeof(y)}(k, x, y)
    end
end
# with euclidean dot product
Gramian(x::AbstractVector, y::AbstractVector) = Gramian(Kernel.Dot(), x, y)

# TODO: + HODLR?
function LinearAlgebra.factorize(G::Gramian; check::Bool = true, tol::Real = 1e-12)
    cholesky(G, Val(true); check = check, tol = tol) # my own implementation of pivoted cholesky
end

# TODO: could define algebra
# import Base:+
# +(G::Gramian, H::Gramian) = (G.x ≡ H.x && G.y ≡ H.y) ? Gramian(G.k+H.k, x, y) :
#                     error("cannot add two Gramian matrices with different x or y")

import Base: size, getindex
size(K::Gramian) = (length(K.x), length(K.y))

function getindex(G::Gramian, i::Integer, j::Integer)
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds G.k(G.x[i], G.y[j]) # remove boundscheck of x of x and y
end

# TODO: should we make this a view?
function getindex(G::Gramian, i::Union{AbstractArray, Colon},
                                                j::Union{AbstractArray, Colon})
    @boundscheck checkbounds(G, i, j) # add bounds check to G
    @inbounds gramian(G.k, G.x[i], G.y[j])
end

Base.eltype(G::Gramian{T}) where {T} = T

import LinearAlgebra: issymmetric, isposdef, ishermitian
# this yields the incorrect result in the rare case that x and y are identical,
# but are not stored in the same place in memory. The benefit of this is
# that is can be computed in constant time
issymmetric(G::Gramian) = G.x === G.y
isposdef(G::Gramian) = issymmetric(G)
ishermitian(G::Gramian) = issymmetric(G)

######################### smart pseudo-constructor #############################
# standard approach is a lazily represented kernel matrix
gramian(k::MercerKernel, x::AbstractVector, y::AbstractVector) = Gramian(k, x, y)
gramian(k::MercerKernel, x) = gramian(k, x, x)

gramian(x::AbstractVector, y::AbstractVector) = Gramian(x, y)
gramian(x::AbstractVector) = gramian(x, x)

# if matrix whose columns are datapoints is passed, convert to vector of vectors
gramian(k::MercerKernel, x::AbstractMatrix) = gramian(k, vecofvec(x))
function gramian(k::MercerKernel, x::AbstractMatrix, y::AbstractMatrix)
    gramian(k, vecofvec(x), vecofvec(y))
end

# to project vectors into the RKHS formed by k on x
import LinearAlgebraExtensions: Projection
Projection(k::MercerKernel, x::AbstractVector) = Projection(gramian(k, x))

# this has to be done lazily to preserve structure
# gramian(k::VerticalRescaling, x) = Diagonal(k.f.(x)) * marginal(k.k, x) * Diagonal(k.f.(x))
# only be lazy if there is special structure (StepRangeLen)
# function gramian(k::VerticalRescaling{<:Real, <:StationaryKernel}, x::StepRangeLen)
#     LazySymmetricRescaling(Diagonal(k.a.(x)), marginal(k.k, x))
# end

# could have lazy ones matrix, not sure if relevant
# gramian(k::Constant, x) = k.c * ones(eltype(x), (length(x), length(x))) # fill array?

# 1D statonary kernel on equi-spaced grid gives rise to toeplitz structure
# have to change the cholesky call
# function gramian(k::StationaryKernel, x::StepRangeLen{<:Real}, ::Val{false} = Val(false))
#     SymmetricToeplitz(k.(x[1], x)) # return as CovarianceMatrix?
# end

# import Base: +
# function +(T::SymmetricToeplitz, D::UniformScaling)
#     vc = copy(T.vc)
#     vc[1] += D.λ
#     SymmetricToeplitz(vc)
# end

# gramian(k::StationaryKernel, x::StepRangeLen{<:Real}, y::StepRangeLen{<:Real},
#                     ::Val{false} = Val(false)) = SymmetricToeplitz(k.(x[1], y))


# 1D stationary kernel on equi-spaced grid with periodic boundary conditions
gramian(k::StationaryKernel, x::StepRangeLen{<:Real},
                            ::Val{true}) = Circulant(k.(x[1], x))
