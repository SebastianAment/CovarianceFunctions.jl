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

# abstract type Kernel{T, S} end # could have "isscalar" field
abstract type AbstractKernel{T} end # could be defined by input and output type ...
abstract type MercerKernel{T} <: AbstractKernel{T} end
# TODO: these should be traits
abstract type StationaryKernel{T} <: MercerKernel{T} end
abstract type IsotropicKernel{T} <: StationaryKernel{T} end # ίσος + τρόπος (equal + way)

# class of matrix-valued kernels for multi-output GPs
abstract type MultiKernel{T} <: AbstractKernel{T} end # MultiKernel

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

# lazy way to represent gramian matrices
# note: gramian specializations for special matrix structure
# has to be after definition of all kernels
include("gramian.jl") # deprecate in favor of kernel matrix?
# include("kernel_matrix.jl")

include("gradient.jl")
include("gpkernels.jl")
include("properties.jl")

import LinearAlgebraExtensions: iscov
iscov(k::MercerKernel, x = randn(32), tol = 1e-10) = iscov(gramian(k, x), tol)

############################# covariance kernel functions ######################
Statistics.cov(k::MercerKernel, x::AbstractVector) = gramian(k, x)

end # Kernel
