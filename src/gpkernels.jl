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

###################### Sparse Approximation Kernels ############################
#################### Subset of Regressors (SoR) kernel #########################
# TODO:
# SoR can suffer from non-sensical predictive variances
# To avoid this, augment inducing variables by points close to the predictive points
# use this to implement "Sparse Greedy Gaussian Process Regression" and SKI
struct SubsetOfRegressors{T, K<:MercerKernel{T}, U,
                        S<:AbstractMatOrFac} <: MercerKernel{T}
    k::K
    xᵤ::U # inducing inputs
    Σᵤᵤ::S # covariance between inducing variables
end
const SoR = SubsetOfRegressors

# we probably don't even need to factorize the covariance because of the Woodbury
function SoR(k::MercerKernel, xᵤ::AbstractVector)
    # SoR(k, xᵤ, factorize(gramian(k, xᵤ)))
    SoR(k, xᵤ, factorize(gramian(k, xᵤ)))
end

# this call won't be efficient, if we allow scalar -> overload cov / marginal
# avoid this at all cost!
function (k::SoR)(x, y)
    dot(gramian(k.k, [x], k.xᵤ), inverse(k.Σᵤᵤ), gramian(k.k, k.xᵤ, [y]))
end

# TODO: might have to cast gramian to Matrix for better performance
function gramian(k::SoR, x::AbstractVector, y::AbstractVector)
    Σᵤₓ = gramian(k.k, k.xᵤ, x) # cross covariance
    Σᵤᵥ = (x ≡ y) ? Σᵤₓ : gramian(k.k, k.xᵤ, y) # change when lowercase unicode y comes in
    Σᵤᵤ = factorize(k.Σᵤᵤ)
    *(Σᵤₓ', inverse(Σᵤᵤ), Σᵤᵥ)
    #     U = Σᵤᵤ.U \ Σᵤₓ
    #     SymmetricLowRank(U) ?
end

# specializing factorize for SubsetOfRegressors Kernel
# function factorize(K::Gramian{<:Real, <:SoR}, like::Diagonal)
#     return -1
# end

# computational complexity of woodbury is
# O(m^3) + O(n m^2) < O(n^3)? => 1 + (n/m) < (n/m)^3 ?
# x^3 - x = x (x^2-1) = 1 -> 1.3247
# TODO: we need to optimize for c, so that woodbury is only employed
# when it is efficient
# TODO: this can be generalized if we specialize factorize on the SoR kernel matrix
# (Kyu Kuu Kux) (Kxu Kuu Kux + I)^{-1} (Kxu Kuu Kuy)
# using CovarianceMatrices: covariance
# function posterior(k::SoR, like::AbstractMatOrFac, x::AbstractVector, c::Int = 2)
#     Kᵤₓ = gramian(k.k, k.xᵤ, x)
#     Σₓₓ = Woodbury(like, Kᵤₓ', k.Σᵤᵤ, Kᵤₓ)
#     # potentially add covariance cast?
#     # Σₓₓ = covariance(Σₓₓ)
#     Σₓₓ = factorize(Σₓₓ)
#     ConditionalKernel(k, x, Σₓₓ)
#     # for efficiency: convert back to SoR kernel, need lazy matrix difference
#     # S = inverse(k.Σᵤᵤ) * Kᵤₓ
#     # Σᵤᵤ = *(S, Σₓₓ, S')
#     # SoR(k, x, k.Σᵤᵤ - Σᵤᵤ)
# end

# function posterior(k::SoR, σ²::Real, x::AbstractVector, c::Int = 2)
#     σ² ≤ 0 && error("σ² ≤ 0")
#     like = (σ²*I)(length(x)) # noise variance
#     posterior(k, like, x, c)
# end


# function marginal(k::SoR, x::AbstractVector, σ²::Real)
#     σ² ≤ 0 && error("σ² ≤ 0")
#     D = (σ²*I)(length(k.xᵤ)) # noise variance
#     Σᵤₓ = k.(k.xᵤ, permutedims(x)) #
#     SymWoodbury(D, Σᵤₓ', inverse(k.Σᵤᵤ))
# end

# ################ Deterministic Training Conditional (SoD) kernel ###############
# does not correspond to a true GP! Therefore, I am not implementing it.
# Also, it is another headache because we would need to specialize the conditioning operation,
# because the predictive variance is calculated differently to the test variance
# struct DTC{T, K<:MercerKernel, U} <: MercerKernel
#     k::K
#     xᵤ::U # inducing inputs
#     Σᵤᵤ::U # factorized covariance between inducing variables
# end
