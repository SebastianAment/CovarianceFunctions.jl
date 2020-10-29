############################ stationary kernels ################################
using LinearAlgebraExtensions: ispsd, difference

# notation:
# x, y inputs
# τ = x-y difference
(k::MercerKernel)(x, y) = k(difference(x, y)) # if the two argument signature is not defined, must be stationary
(k::MercerKernel)(τ) = k(norm(τ)) # if only the scalar argument form is defined, must be isotropic

############################# constant kernel ##################################
# can be used to rescale existing kernels
# TODO: Allow Matrix valued constant
struct Constant{T} <: IsotropicKernel{T}
    c::T
    function Constant(c, check::Bool = true)
        if check && !ispsd(c)
            throw(DomainError("Constant is not positive semi-definite: $c"))
        end
        new{typeof(c)}(c)
    end
end
# Constant(c) = Constant{typeof(c)}(c)
parameters(k::Constant) = [k.c]
nparameters(::Constant) = 1

# should type of constant field and τ agree? what promotion is necessary?
# do we need the isotropic/ stationary evaluation, if we overwrite the mercer one?
(k::Constant)(τ) = k.c # stationary / isotropic
(k::Constant)(x, y) = k.c # mercer

#################### standard exponentiated quadratic kernel ###################
struct ExponentiatedQuadratic{T} <: IsotropicKernel{T} end
const EQ = ExponentiatedQuadratic
EQ() = EQ{Float64}()

(k::EQ)(τ::Real) = exp(-τ^2/2)

########################## rational quadratic kernel ###########################
struct RationalQuadratic{T} <: IsotropicKernel{T}
    α::T # relative weighting of small and large length scales
    RationalQuadratic{T}(α) where T = (0 < α) ? new(α) : throw(DomainError("α not positive"))
end
const RQ = RationalQuadratic
RQ(α::Real) = RQ{typeof(α)}(α)

(k::RQ)(τ::Real) = (1 + τ^2 / (2*k.α))^-k.α
parameters(k::RQ) = [k.α]
nparameters(::RQ) = 1

########################### exponential kernel #################################
struct Exponential{T} <: IsotropicKernel{T} end
const Exp = Exponential
Exp() = Exp{Float64}()

(k::Exp)(τ::Real) = exp(-abs(τ))

############################ γ-exponential kernel ##############################
struct GammaExponential{T<:Real} <: IsotropicKernel{T}
    γ::T
    GammaExponential{T}(γ) where {T} = (0 ≤ γ ≤ 2) ? new(γ) : throw(DomainError("γ not in [0,2]"))
end
const γExp = GammaExponential
γExp(γ::T) where T = γExp{T}(γ)

(k::γExp)(τ::Real) = exp(-abs(τ)^(k.γ)/2)
parameters(k::γExp) = [k.γ]
nparameters(::γExp) = 1

########################### white noise kernel #################################
struct Delta{T} <: IsotropicKernel{T} end
const δ = Delta
δ() = δ{Float64}()
# TODO: returning type T might not work well with A.D., should it?
(k::δ{T})(τ) where {T} = T(all(x->isequal(x, 0), τ))
(k::δ{T})(x, y) where {T} = T(all(x -> isequal(x...), zip(x, y)))
gramian(::δ{T}, x::AbstractVector) where {T} = (one(T)*I)(length(x))

############################ Matern kernel #####################################
# TODO: use rational types to dispatch to MaternP evaluation, i.e. 5//2 -> MaternP(3)
# seems k/2 are representable exactly in floating point?
struct Matern{T} <: IsotropicKernel{T}
    ν::T
    Matern{T}(ν) where T = (0 < ν) ? new(ν) : throw(DomainError("ν = $ν is negative"))
end
Matern(ν::T) where {T} = Matern{T}(ν)
parameters(k::Matern) = [k.ν]
nparameters(::Matern) = 1

# IDEA: could have value type argument to dispatch p parameterization
function (k::Matern)(τ::Real)
    if τ ≈ 0
        one(τ)
    else
        ν = k.ν
        r = sqrt(2ν) * abs(τ)
        2^(1-ν) / gamma(ν) * r^ν * besselk(ν, r)
    end
end

################# Matern kernel with ν = p + 1/2 where p ∈ ℕ ###################
struct MaternP{T} <: IsotropicKernel{T}
    p::Int
    MaternP{T}(p::Int) where T = 0 ≤ p ? new(p) : throw(DomainError("p = $p is negative"))
end

MaternP(p::Int = 0) = MaternP{Float64}(p)
MaternP(k::Matern) = MaternP(floor(Int, k.ν)) # project Matern to closest MaternP

function (k::MaternP)(τ::Real)
    p = k.p
    val = zero(τ)
    r = sqrt(2p+1) * abs(τ)
    for i in 0:p
        val += (factorial(p+i)/(factorial(i)*factorial(p-i))) * (2r)^(p-i)
    end
    val *= exp(-r) * (factorial(p)/factorial(2p))
end

########################### cosine kernel ######################################
# interesting because it allows negative co-variances
# it is a valid stationary kernel,
# because it is the inverse Fourier transform of point measure at μ (delta distribution)
struct CosineKernel{T, V<:Union{T, AbstractVector{T}}} <: StationaryKernel{T}
    μ::V
end
const Cosine = CosineKernel
# IDEA: trig-identity -> low-rank gramian
(k::CosineKernel)(τ) = cos(2π * dot(k.μ, τ))
(k::CosineKernel{<:Real, <:Real})(τ) = cos(2π * k.μ * sum(τ))

parameters(k::CosineKernel) = k.μ isa Real ? [k.μ] : k.μ
nparameters(k::CosineKernel) = length(k.μ)

####################### spectral mixture kernel ################################
# can be seen as product kernel of Constant, Cosine, ExponentiatedQuadratic
Spectral(w::Real, μ, l) = prod((w, Cosine(μ), ARD(EQ(), l))) # 2π^2/σ.^2)
SpectralMixture(w::AbstractVector, μ, l) = sum(Spectral.(w, μ, l))
const SM = SpectralMixture

############################ Cauchy Kernel #####################################
# there is something else in the literature with the same name ...
struct Cauchy{T} <: IsotropicKernel{T} end
Cauchy() = Cauchy{Float64}()
(k::Cauchy)(τ::Real) =  1/(π * (1+τ^2))

# for spectroscopy
PseudoVoigt(α::T) where T<:Real = α*EQ{T}() + (1-α)*Cauchy{T}()

###################### Inverse Multi-Quadratic  ################################
# seems formally similar to Cauchy, Cauchy is equal to power of IMQ
struct InverseMultiQuadratic{T} <: IsotropicKernel{T}
    c::T
end
(k::InverseMultiQuadratic)(τ::Real) = 1/√(τ^2 + k.c^2)
