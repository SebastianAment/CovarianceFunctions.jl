# Kernel properties
ismercer(::T) where T = false
ismercer(::MercerKernel) = true

isstationary(::T) where T = false
isstationary(::StationaryKernel) = true
isstationary(::IsotropicKernel) = true

isisotropic(::T) where T = false # default to false
isisotropic(::StationaryKernel) = false
isisotropic(::IsotropicKernel) = true

isdot(::T) where T = false # dot product kernel
isdot(::Union{Dot, ExponentialDot}) = true

# TODO: check if this is doing constant propagation
const ProductsAndSums{T, AT} = Union{Sum{T, AT}, Product{T, AT},
                                SeparableProduct{T, AT}, SeparableSum{T, AT}}
ismercer(S::ProductsAndSums) = all(ismercer, S.args)
ismercer(P::Power) = ismercer(P.k)
# TODO: this needs special treatement for constant kernels, which can be all input types
isstationary(S::ProductsAndSums) = all(isstationary, S.args)
isstationary(P::Power) = isstationary(P.k)

isisotropic(S::ProductsAndSums) = all(isisotropic, S.args)
isisotropic(P::Power) = isisotropic(P.k)

isdot(S::ProductsAndSums) = all(isdot, S.args) # need to special case constant kernel
isdot(P::Power) = isdot(P.k)

abstract type InputTrait end
struct GenericInput <: InputTrait end
struct IsotropicInput <: InputTrait end
struct DotProductInput <: InputTrait end
struct StationaryInput <: InputTrait end

input_trait(::T) where T = GenericInput()
input_trait(::StationaryKernel) = StationaryInput()
input_trait(::IsotropicKernel) = IsotropicInput()
input_trait(::Union{Dot, ExponentialDot}) = DotProductInput()
input_trait(P::Power) = input_trait(P.k)

# special treatment for constant kernel, since it can function for any input
function input_trait(S::ProductsAndSums)
    i = findfirst(x->!isa(x, Constant), S.args)
    if isnothing(i) # all constant kernels
        return IsotropicInput()
    else
        trait = input_trait(S.args[i]) # first non-constant kernel
        for j in i+1:length(S.args)
            k = S.args[j]
            if k isa Constant
                continue
            elseif input_trait(k) != trait # if the non-constant kernels don't have the same input type,
                return GenericInput() # we default back to GenericInput
            end
        end
        return trait
    end
end

# traits (works on type level only)
# mercer_trait(::T) where T = Val(false)
# stationary_trait(::T) where T = Val(false)
# isotropic_trait(::T) where T = Val(false)
# dot_trait(::T) where T = Val(false)
#
# mercer_trait(::Type{<:MercerKernel}) = Val(true)
# stationary_trait(::Type{<:StationaryKernel}) = Val(true)
# isotropic_trait(::Type{<:IsotropicKernel}) = Val(true)
# dot_trait(::Type{<:Dot}) = Val(true)
#
# mercer_trait(P::Power) = mercer_trait(P.k)
# stationary_trait(P::Power) = stationary_trait(P.k)
# isotropic_trait(P::Power) = isotropic_trait(P.k)
# dot_trait(P::Power) = dot_trait(P.k)

# TODO:
# _tuple_trait(trait, ::Type{Tuple{T, Vararg}}) where T = (trait(T),
# function _product_sum_trait(trait, S::Type{<:ProductsAndSums{T, AT}}) where {T, AT}
#     trait.(AT) isa NTuple ? Val(true) : Val(false)
# end
# mercer_trait(S::Type{<:ProductsAndSums}) = _product_sum_trait(mercer_trait, S)
# stationary_trait(S::Type{<:ProductsAndSums}) = _product_sum_trait(stationary_trait, S)
# isotropic_trait(S::Type{<:ProductsAndSums}) = _product_sum_trait(isotropic_trait, S)
# dot_trait(S::Type{<:ProductsAndSums}) = _product_sum_trait(dot_trait, S)
