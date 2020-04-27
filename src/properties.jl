# Kernel properties
const IsotropicKernels = Union{Constant, EQ, RQ, Exp, γExp, δ, Matern, MaternP,
    Cauchy, InverseMultiQuadratic, Lengthscale, Periodic}

const StationaryKernels = Union{IsotropicKernels, Cosine, Normed}

# Does it work if these are values via constant propagation?
isisotropic(::AbstractKernel) = false
isstationary(::AbstractKernel) = false

isisotropic(::StationaryKernels) = false
isstationary(::StationaryKernels) = true

isisotropic(::IsotropicKernels) = true
isstationary(::IsotropicKernels) = true

const ProductsAndSums = Union{Sum, Product, SeparableProduct, SeparableSum}
isstationary(S::ProductsAndSums) = all(isstationary, S.args)
isisotropic(S::ProductsAndSums) = all(isisotropic, S.args)
isisotropic(P::Power) = isisotropic(P.k)
isstationary(P::Power) = isstationary(P.k)
