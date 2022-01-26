module CovarianceFunctions

export AbstractKernel, MercerKernel, StationaryKernel, IsotropicKernel, MulitKernel
export gramian, Gramian
export DotProductInput, IsotropicInput, GenericInput, InputTrait, input_trait

using LinearAlgebra
using SparseArrays
using StaticArrays
using FillArrays
using LazyArrays
using Base.Threads

using ForwardDiff
using DiffResults

using SpecialFunctions: gamma, besselk

using KroneckerProducts
using WoodburyFactorizations
using BlockFactorizations
using NearestNeighbors
using IterativeSolvers

abstract type AbstractKernel{T} end
abstract type MercerKernel{T} <: AbstractKernel{T} end
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

const default_tol = 1e-6 # default tolerance for matrix solves and products

# IDEA: rewrite stationary kernels as function of r² to avoid AD problems

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} <: AbstractKernel{T} end # MultiKernel
# compute the (i, j)th entry of k(x, y)
Base.getindex(K::MultiKernel, i, j) = (x, y) -> getindex(K(x, y), i, j)

# first, utility functions
include("util.jl")
include("lazy_linear_algebra.jl") # TODO: separate out into package
include("lazy_grid.jl")
include("toeplitz.jl") # special functions for toeplitz matrices -> put in ToeplitzMatrices.jl

include("derivatives.jl")
include("parameters.jl")

# include all types of kernels
include("algebra.jl") # kernel operations
include("stationary.jl") # stationary mercer kernels
include("transformation.jl")
include("mercer.jl") # general mercer kernels

include("properties.jl")
include("gramian.jl")
include("sparse.jl")
include("barneshut.jl")

# including multi-output kernels
include("gradient.jl")
include("gradient_algebra.jl")
include("hessian.jl")
include("separable.jl")

end # CovarianceFunctions
