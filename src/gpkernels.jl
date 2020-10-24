########################## Gaussian Process Kernels ############################
using LazyInverse
using WoodburyIdentity

# TODO: use UniformScaling instead of conversion to Diagonal for likelihood
########################## conditional kernel ##################################
# Important: x has to be a vector of data, even if it is just a single element
# results from conditing gp on data occuring at x
struct ConditionalKernel{T, K<:MercerKernel{T}, X<:AbstractVector,
                    S<:Union{Factorization, AbstractMatrix}} <: MercerKernel{T}
    k::K
    x::X
    Σₓₓ::S # covariance between x
end

# Σₓₓ might be constructed based on kernel type
function ConditionalKernel(k::MercerKernel, x::AbstractVector)
    Σₓₓ = factorize(gramian(k, x))
    ConditionalKernel(k, x, Σₓₓ)
end
conditional(k::MercerKernel, x) = ConditionalKernel(k, x)

# might have to force @nospecialize for K, since recursive Conditional usage
# would be stressing the compiler too much
function (k::ConditionalKernel)(x, y)
    Σᵤₓ = gramian(k.k, k.x, [x])
    Σᵤᵥ = (x ≡ y) ? Σᵤₓ : gramian(k.k, k.x, [y])
    k.k(x, y) - dot(Σᵤₓ, inverse(k.Σₓₓ), Σᵤᵥ)
end

######################### with noisy observations ##############################
# TODO: could define conditional(k, x, y, σ = 0.) to subsume posterior
# calculate posterior kernel
function posterior(k::MercerKernel, like::AbstractMatOrFac, x::AbstractVector)
    Σₓₓ = factorize(Matrix(gramian(k, x) + like))
    ConditionalKernel(k, x, Σₓₓ)
end

# Conveniences for homoscedastic and heteroscedastic likelihoods
function posterior(k::MercerKernel, σ²::Real, x::AbstractVector)
    σ² == 0 && conditional(k, x)
    σ² < 0 && throw(DomainError("σ² < 0"))
    posterior(k, (σ²*I)(length(x)), x)
end

# use Woodbury identity for Diagonal noise covariance
function posterior(k::MercerKernel, Σ::Diagonal, x::AbstractVector)
    F = factorize(gramian(k, x))
    Σₓₓ = F isa LowRank ? Woodbury(Σ, F) : Σ + Matrix(F)
    Σₓₓ = factorize(Σₓₓ)
    ConditionalKernel(k, x, Σₓₓ)
end

function posterior(k::MercerKernel, σ²::Function, x::AbstractVector)
    posterior(k, Diagonal(σ².(x)), x)
end

function posterior(k::MercerKernel, σ²::Real, x::AbstractVector,
                                                ::Val{true}, c::Real = 2)
    σ² ≤ 0 && error("σ² ≤ 0")
    n = length(x)
    like = (σ²*I)(n) # noise variance
    # Kₓₓ = gramian(k, x)
    Kₓₓ = marginal(k, x)
    Kₓₓ = factorize(Kₓₓ) # could restrict rank here, call cholesky directly?
    Σₓₓ = Woodbury(like, Kₓₓ.U', I(Kₓₓ.rank), Kₓₓ.U) # this requires Cholesky, or SymmetricLowRank
    # TODO: need to insert covariance cast?
    Σₓₓ = factorize(Σₓₓ)
    ConditionalKernel(k, x, Σₓₓ)
end

# this might dispatch on the type of the factorization to be efficient
# does it make sense to return a lazy matrix? given each element requires O(n^2) cost ...
# specialize Gramian for better complexity
function gramian(k::ConditionalKernel, x::AbstractVector, y::AbstractVector)
    Σᵤₓ = gramian(k.k, k.x, x)
    Σᵤᵥ = (x ≡ y) ? Σᵤₓ : gramian(k.k, k.x, y)
    # TODO: convert to Woodbury call, does this allow us to specialize
    # ternary * for Woodbury as middle argument, speeding up SoR?
    # Σ = gramian(k.k, x, y) - *(Σᵤₓ', inverse(k.Σₓₓ), Σᵤᵥ)
    Σ = Woodbury(gramian(k.k, x, y), Σᵤₓ', inverse(k.Σₓₓ), Σᵤᵥ, -)
    return Σ
end
