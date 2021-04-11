module CovarianceFunctions

export AbstractKernel, MercerKernel, StationaryKernel, IsotropicKernel, MulitKernel
export gramian, Gramian

using LinearAlgebra
using SparseArrays
using FillArrays
using LazyArrays
using Base.Threads

using ForwardDiff
using DiffResults

using SpecialFunctions: gamma, besselk

# type unions we need often:
using LinearAlgebraExtensions
using LinearAlgebraExtensions: AbstractMatOrFac
using KroneckerProducts
using WoodburyIdentity

const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}

abstract type AbstractKernel{T} end
abstract type MercerKernel{T} <: AbstractKernel{T} end
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} <: AbstractKernel{T} end # MultiKernel
# compute the (i, j) entry of k(x, y)
Base.getindex(K::MultiKernel, i::Integer, j::Integer) = (x, y) -> K(x, y)[i, j]

# first, utility functions
include("util.jl")
include("derivatives.jl")
include("block.jl")
include("parameters.jl")

# include all types of kernels
include("algebra.jl") # kernel operations
include("stationary.jl") # stationary mercer kernels
include("transformation.jl")
include("mercer.jl") # general mercer kernels

include("properties.jl")
include("gramian.jl") # deprecate in favor of kernel matrix?

# including multi-output kernels
include("gradient.jl")
include("hessian.jl")
include("separable.jl")

end # Kernel
