############################ stationary kernels ################################
# using LinearAlgebraExtensions: difference
# notation:
# x, y inputs
# τ = x-y difference of inputs
# r = norm(x-y) norm of τ
(k::MercerKernel)(x, y) = k(norm(difference(x, y))) # if the two argument signature is not defined, default to stationary
(k::MercerKernel)(r) = k(norm(r)) # if only the scalar argument form is defined, must be isotropic

############################# constant kernel ##################################
# can be used to rescale existing kernels
# IDEA: Allow Matrix-valued constant
struct Constant{T} <: IsotropicKernel{T}
    c::T
    function Constant(c, check::Bool = true)
        if check && !ispsd(c)
            throw(DomainError("Constant is not positive semi-definite: $c"))
        end
        new{typeof(c)}(c)
    end
end
# isisotropic(::Constant) = true
# ismercer(k::Constant) = ispsd(k.c)
# Constant(c) = Constant{typeof(c)}(c)
parameters(k::Constant) = [k.c]
nparameters(::Constant) = 1

# should type of constant field and r agree? what promotion is necessary?
# do we need the isotropic/ stationary evaluation, if we overwrite the mercer one?
(k::Constant)(r) = k.c # stationary / isotropic
(k::Constant)(x, y) = k.c # mercer

#################### standard exponentiated quadratic kernel ###################
struct ExponentiatedQuadratic{T} <: IsotropicKernel{T} end
const EQ = ExponentiatedQuadratic
EQ() = EQ{Union{}}() # defaults to "bottom" type since it doesn't have any parameters

(k::EQ)(r::Number) = exp(-r^2/2)

########################## rational quadratic kernel ###########################
struct RationalQuadratic{T} <: IsotropicKernel{T}
    α::T # relative weighting of small and large length scales
    RationalQuadratic{T}(α) where T = (0 < α) ? new(α) : throw(DomainError("α not positive"))
end
const RQ = RationalQuadratic
RQ(α::Real) = RQ{typeof(α)}(α)

(k::RQ)(r::Number) = (1 + r^2 / (2*k.α))^-k.α

parameters(k::RQ) = [k.α]
nparameters(::RQ) = 1

########################### exponential kernel #################################
struct Exponential{T} <: IsotropicKernel{T} end
const Exp = Exponential
Exp() = Exp{Union{}}()

(k::Exp)(r::Number) = exp(-r)

############################ γ-exponential kernel ##############################
struct GammaExponential{T<:Real} <: IsotropicKernel{T}
    γ::T
    GammaExponential{T}(γ) where {T} = (0 ≤ γ ≤ 2) ? new(γ) : throw(DomainError("γ not in [0,2]"))
end
const γExp = GammaExponential
γExp(γ::T) where T = γExp{T}(γ)

(k::γExp)(r::Number) = exp(-r^k.γ / 2)
parameters(k::γExp) = [k.γ]
nparameters(::γExp) = 1

########################### white noise kernel #################################
struct Delta{T} <: IsotropicKernel{T} end
const δ = Delta
δ() = δ{Union{}}()

(k::δ)(r) = all(iszero, r) ? one(eltype(r)) : zero(eltype(r))
function (k::δ)(x, y)
    T = promote_type(eltype(x), eltype(y))
    (x == y) ? one(T) : zero(T) # IDEA: if we checked (x === y) could incorporate noise variance for vector inputs -> EquivDelta?
end
############################ Matern kernel #####################################
# IDEA: use rational types to dispatch to MaternP evaluation, i.e. 5//2 -> MaternP(3)
# seems k/2 are representable exactly in floating point?
struct Matern{T} <: IsotropicKernel{T}
    ν::T
    Matern{T}(ν) where T = (0 < ν) ? new(ν) : throw(DomainError("ν = $ν is negative"))
end
Matern(ν::T) where {T} = Matern{T}(ν)
parameters(k::Matern) = [k.ν]
nparameters(::Matern) = 1

# IDEA: could have value type argument to dispatch p parameterization
function (k::Matern)(r::Number)
    if r ≈ 0
        one(r)
    else
        ν = k.ν
        r *= sqrt(2ν)
        2^(1-ν) / gamma(ν) * r^ν * besselk(ν, r)
    end
end

################# Matern kernel with ν = p + 1/2 where p ∈ ℕ ###################
struct MaternP{T} <: IsotropicKernel{T}
    p::Int
    MaternP{T}(p::Int) where T = 0 ≤ p ? new(p) : throw(DomainError("p = $p is negative"))
end

MaternP(p::Int = 0) = MaternP{Union{}}(p)
MaternP(k::Matern) = MaternP(floor(Int, k.ν)) # project Matern to closest MaternP

function (k::MaternP)(r::Number)
    p = k.p
    val = zero(r)
    r *= sqrt(2p+1)
    for i in 0:p
        val += (factorial(p+i)/(factorial(i)*factorial(p-i))) * (2r)^(p-i) # putting @fastmath here leads to NaN with ForwardDiff
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
(k::CosineKernel)(r) = cos(2π * dot(k.μ, r))
(k::CosineKernel{<:Real, <:Real})(r) = cos(2π * k.μ * sum(r))

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
Cauchy() = Cauchy{Union{}}()
(k::Cauchy)(r::Number) = inv(1+r^2) # π is not necessary, we are not normalizing

# for spectroscopy
PseudoVoigt(α) = α*EQ() + (1-α)*Cauchy()

###################### Inverse Multi-Quadratic  ################################
# seems formally similar to Cauchy, Cauchy is equal to power of IMQ
struct InverseMultiQuadratic{T} <: IsotropicKernel{T}
    c::T
end
(k::InverseMultiQuadratic)(r::Number) = 1/√(r^2 + k.c^2)
