# CovarianceFunctions.jl 
[![CI](https://github.com/SebastianAment/CovarianceFunctions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianAment/CovarianceFunctions.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/SebastianAment/CovarianceFunctions.jl/branch/main/graph/badge.svg?token=04NSNJC9H1)](https://codecov.io/gh/SebastianAment/CovarianceFunctions.jl)

This package implements many popularly used Mercer kernels.
Commonly used abstract types are
`IsotropicKernel <: StationaryKernel <: MercerKernel`.
A `MercerKernel` satisfies the Mercer axioms and implements
`function (k::AMercerKernel)(x::T, y::T) where {T} end`.

A `StationaryKernel` implements
`function (k::AStationaryKernel)(τ) end`,
where τ = x-y, which can potentially be a vector.

Lastly, a `IsotropicKernel` only implements the one argument function signature
where the argument is restricted to be real.
This is because isotropy implies that the kernel is only a function of |τ|.

In contrast to other packages with a similar purpose,
this package implements a lazy `Gramian` type, which
can be used to solve many linear algebraic problems arising in
kernel methods efficiently.
Importantly, the code recognizes when linear algebraic structure
is present and exploits it for computational efficiency.

To extend this package with new kernels, we just need to define a kernel type:

```
struct MyKernel{T} <: MercerKernel{T} end
(k::NewKernel)(x, y) = ...
```

# TODO:
* write KernelFamily struct, which consists of a parametric family of kernels
* write efficient gradient of kernel w.r.t. inputs

* more covariance matrix factorizations (HODLR), or approximations (SKI, Nystrom, Random Kitchen Sinks, Fast Food)
* GPLVMs
* efficient conditional kernel, probably will have to be via functional basis
* more preallocation

* chebyshev interpolation
* functional eigendecomposition of kernels as pre-processing step

* potentially have Degenerate as abstract type or trait, e.g. for polynomial kernel,
cosine kernel etc.

* `const SmoothKernels = Union{EQ, RQ} -> trylowrank(::SmoothKernels, n::Int) = n > 128`
