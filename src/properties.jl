# Kernel properties
isisotropic(::AbstractKernel) = false
isstationary(::AbstractKernel) = false

isisotropic(::StationaryKernel) = false
isstationary(::StationaryKernel) = true

isisotropic(::IsotropicKernel) = true
isstationary(::IsotropicKernel) = true

const ProductsAndSums = Union{Sum, Product, SeparableProduct, SeparableSum}
isstationary(S::ProductsAndSums) = all(isstationary, S.args)
isisotropic(S::ProductsAndSums) = all(isisotropic, S.args)
isisotropic(P::Power) = isisotropic(P.k)
isstationary(P::Power) = isstationary(P.k)
