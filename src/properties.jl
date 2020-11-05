# Kernel properties
isisotropic(::AbstractKernel) = false # default to false
isstationary(::AbstractKernel) = false

isisotropic(::StationaryKernel) = false
isstationary(::StationaryKernel) = true

isisotropic(::IsotropicKernel) = true
isstationary(::IsotropicKernel) = true

ismercer(::MercerKernel) = true

const ProductsAndSums = Union{Sum, Product, SeparableProduct, SeparableSum}
ismercer(S::ProductsAndSums) = all(ismercer, S.args)
ismercer(P::Power) = ismercer(P.k)

isstationary(S::ProductsAndSums) = all(isstationary, S.args)
isstationary(P::Power) = isstationary(P.k)

isisotropic(S::ProductsAndSums) = all(isisotropic, S.args)
isisotropic(P::Power) = isisotropic(P.k)
