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
using LinearAlgebraExtensions: AbstractMatOrFac

# abstract type Kernel{T, S} end # could have "isscalar" field
abstract type AbstractKernel{T} end # could be defined by input and output type ...
abstract type MercerKernel{T} <: AbstractKernel{T} end
# TODO: these should be traits
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} end # <: AbstractKernel{AbstractMatrix{T}} end # MultiKernel

const AllKernels{T} = Union{MercerKernel{T}, MultiKernel{T}}

################################################################################
# parameters function returns either scalar or recursive vcat of parameter vectors
# useful for optimization algorithms which require the parameters in vector form
parameters(::Any) = []
parameters(::AbstractKernel{T}) where {T} = zeros(T, 0)
nparameters(::Any) = 0

# checks if θ has the correct number of parameters to initialize a kernel of typeof(k)
function checklength(k::AbstractKernel, θ::AbstractVector)
    nt = length(θ)
    np = nparameters(k)
    if nt ≠ np
        throw(DimensionMismatch("length(θ) = $nt ≠ $np = nparameters(k)"))
    end
    return nt
end

# thanks to ffevotte in https://discourse.julialang.org/t/how-to-call-constructor-of-parametric-family-of-types-efficiently/38503/5
stripped_type(x) = stripped_type(typeof(x))
stripped_type(typ::DataType) = Base.typename(typ).wrapper

# fallback for zero-parameter kernels
function Base.similar(k::AbstractKernel, θ::AbstractVector)
    n = checklength(k, θ)
    # kernel = eval(Meta.parse(string(typeof(k).name)))
    kernel = stripped_type(k)
    if n == 0
        kernel()
    elseif n == 1
        kernel(θ[1])
    else
        kernel(θ)
    end
end


Base.similar(k::AbstractKernel, θ::Number) = similar(k, [θ])

################################################################################
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
function checklength(x::AbstractArray, y::AbstractArray)
    lx = length(x)
    ly = length(y)
    if lx == ly; throw(DimensionMismatch("length(x) ($lx) ≠ length(y) ($ly)")) end
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
include("gramian.jl") # deprecate in favor of kernel matrix
# include("kernel_matrix.jl")
include("properties.jl")
include("optimization.jl") # nlml optimziation of kernel hyper-parameters

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

## TODO: abstract and general functions for parameter retrieval
# function parameters(k::AbstractKernel)
#     n = nparameters(k)
#     θ = zeros(fieldtype(k), n)
#     i = 1
#     for field in fieldnames(typeof(k))
#         ni = nparameters(k.)
#         θ[i:i+ni]
#     end
#     θ
# end
#
# nparameters(::AbstractFloat) = 1
# nparameters(::Integer) = 0
# nparameters(v::AbstractArray) = length(v)
#
# function nparameters(k::AbstractKernel)
#     n = 0
#     for field in fieldnames(typeof(k))
#         n += length(k.field)
#     end
#     n
# end
