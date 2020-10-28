module Kernel
# could rename "Mercer"

using LinearAlgebra
using ForwardDiff
using DiffResults
using ForwardDiff: derivative, gradient
const FD = ForwardDiff
using LinearAlgebraExtensions
using KroneckerProducts

# TODO maybe these should be extended in GaussianProcess to keep Kernel free
# of probabilistic functions
using Statistics

# type unions we need often:
using LinearAlgebraExtensions: AbstractMatOrFac

abstract type AbstractKernel{T} end
abstract type MercerKernel{T} <: AbstractKernel{T} end
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} <: AbstractKernel{T} end # MultiKernel
# compute the (i, j) entry of k(x, y)
Base.getindex(K::MultiKernel, i::Integer, j::Integer) = (x, y) -> K(x, y)[i, j]

################################################################################
# if we have Matrix valued kernels, this should be different
Base.eltype(k::AbstractKernel{T}) where {T} = T
Base.eltype(::MultiKernel{T}) where {T} = Matrix{T}

# fieldtype of vector space
fieldtype(k::AbstractKernel) = eltype(k)
fieldtype(x) = eltype(x) # base
fieldtype(x::Union{Tuple, AbstractArray}) = fieldtype(eltype(x))
# # Base.eltype(T; recursive::Val{true}) = T == eltype(T) ? T : eltype(T, recursive = Val(true))

include("util.jl")

# include all types of kernels
include("algebra.jl") # kernel operations
include("stationary.jl") # stationary mercer kernels
include("mercer.jl") # general mercer kernels
# include("multi.jl") # multi-output kernels/ kernels for vector-valued functions
# include("physical.jl") # kernels arising from physical differential equations

include("gramian.jl") # deprecate in favor of kernel matrix?
# include("kernel_matrix.jl")

include("gradient.jl")
include("separable.jl")

# include("gpkernels.jl")
include("properties.jl")

end # Kernel
