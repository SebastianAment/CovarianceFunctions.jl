module Kernel
# could rename "Mercer"
using LinearAlgebra
using ForwardDiff
const FD = ForwardDiff
using LinearAlgebraExtensions
# TODO maybe these should be extended in GaussianProcess to keep Kernel free
# of probabilistic functions
import Statistics: cov, var

# my packages
using Metrics

# type unions we need often:
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}

# abstract type Kernel{T, S} end # could have "isscalar" field
abstract type AbstractKernel{T} end # could be defined by input and output type ...
abstract type MercerKernel{T} <: AbstractKernel{T} end
# TODO: these should be traits
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} end # <: AbstractKernel{AbstractMatrix{T}} end # MultiKernel

const AllKernels{T} = Union{MercerKernel{T}, MultiKernel{T}}

# if we have Matrix valued kernels, this should be different
Base.eltype(k::AllKernels{T}) where {T} = T
# fieldtype of vector space
fieldtype(k::AllKernels) = eltype(k)
fieldtype(x) = eltype(x) # base
fieldtype(x::Union{Tuple, AbstractArray}) = fieldtype(eltype(x))
# Base.eltype(T; recursive::Val{true}) = T == eltype(T) ? T : eltype(T, recursive = Val(true))

# TODO: maybe include in and output dimension of kernel in type?
# this makes it easier to type check admissability of input arguments
# and differentiate between data vector or vector of data
# TODO: or instead, write run-time input check
# change to a boundscheck?
function checklength(x, y)
    lx = length(x)
    ly = length(y)
    lx == ly || throw(DimensionMismatch("length(x) ($lx) ≠ length(y) ($ly)"))
    length(x)
end

# include all types of kernels
include("algebra.jl") # kernel operations
include("stationary.jl") # stationary mercer kernels
include("mercer.jl") # general mercer kernels
# include("multi.jl") # multi-output kernels/ kernels for vector-valued functions
# include("physical.jl") # kernels arising from physical differential equations

# lazy way to represent gramian matrices
# note: gramian specializations for special matrix structure
# has to be after definition of all kernels
include("gramian.jl")
# include("autodiff.jl") # autodifferentiation for nlml optimziation

import LinearAlgebraExtensions: iscov
iscov(k::MercerKernel, x = randn(32), tol = 1e-10) = iscov(gramian(k, x), tol)

############################# covariance kernel functions ######################
import Statistics: var, cov, std
cov(k::MercerKernel, x::AbstractVector) = gramian(k, x)

# variance (computes diagonal of covariance), should be specialized where
# more efficient computation is possible
var(k::MercerKernel) = x->abs(k(x, x))
std(k::MercerKernel) = x->sqrt(max(zero(eltype(x)), var(k)(x)))

end # Kernel
