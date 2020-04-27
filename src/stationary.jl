############################ stationary kernels ################################
using LinearAlgebraExtensions: ispsd, difference

const euclidean = Metrics.EuclideanNorm()
# notation:
# x, y inputs
# τ = x-y difference
(k::MercerKernel)(x, y) = k(difference(x, y)) # if the two argument signature is not defined, must be stationary
(k::MercerKernel)(τ) = k(euclidean(τ)) # if only the scalar argument form is defined, must be isotropic

############################# constant kernel ##################################
# can be used to rescale existing kernels
# TODO: Allow Matrix valued constant!
struct Constant{T} <: IsotropicKernel{T}
    c::T
    Constant(c) = ispsd(c) ? new{typeof(c)}(c) : throw(DomainError("Constant is negative"))
end
parameters(k::Constant) = [k.c]
nparameters(::Constant) = 1

# should type of constant field and τ agree? what promotion is necessary?
# do we need the isotropic/ stationary evaluation, if we overwrite the mercer one?
(k::Constant)(τ) = k.c # stationary / isotropic
(k::Constant)(x, y) = k.c # mercer

# useful?
# Base.zero(::AbstractKernel{T}) where {T} = Constant(zero(T))
# Base.one(::AbstractKernel{T}) where {T} = Constant(one(T))

#################### standard exponentiated quadratic kernel ###################
struct ExponentiatedQuadratic{T} <: IsotropicKernel{T} end
const EQ = ExponentiatedQuadratic
EQ() = EQ{Float64}()

(k::EQ)(τ::Number) = exp(-τ^2/2)

########################## rational quadratic kernel ###########################
struct RationalQuadratic{T} <: IsotropicKernel{T}
    α::T # relative weighting of small and large length scales
    RationalQuadratic{T}(α) where T = (0 < α) ? new(α) : throw(DomainError("α not positive"))
end
const RQ = RationalQuadratic
RQ(α::Real) = RQ{typeof(α)}(α)

(k::RQ)(τ::Number) = (1 + τ^2 / (2*k.α))^-k.α
parameters(k::RQ) = [k.α]
nparameters(::RQ) = 1

########################### exponential kernel #################################
struct Exponential{T} <: IsotropicKernel{T} end
const Exp = Exponential
Exp() = Exp{Float64}()

(k::Exp)(τ::Number) = exp(-abs(τ))

############################ γ-exponential kernel ##############################
struct GammaExponential{T<:Real} <: IsotropicKernel{T}
    γ::T
    GammaExponential{T}(γ) where {T} = (0 ≤ γ ≤ 2) ? new(γ) : throw(DomainError("γ not in [0,2]"))
end
const γExp = GammaExponential
γExp(γ::T) where T = γExp{T}(γ)

(k::γExp)(τ::Number) = exp(-abs(τ)^(k.γ)/2)
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
using SpecialFunctions: gamma, besselk
struct Matern{T} <: IsotropicKernel{T}
    ν::T
    Matern{T}(ν) where T = (0 < ν) ? new(ν) : throw(DomainError("ν = $ν is negative"))
end
Matern(ν::T) where {T} = Matern{T}(ν)
parameters(k::Matern) = [k.ν]
nparameters(::Matern) = 1

# TODO: could have value type argument to dispatch p parameterization
function (k::Matern)(τ::Number)
    if τ ≈ 0
        one(τ)
    else
        ν = k.ν
        r = sqrt(2ν) * abs(τ)
        2^(1-ν) / gamma(ν) * r^ν * besselk(ν, r)
    end
end

################# Matern kernel with ν = p + 1/2 where p ∈ ℕ ###################
# TODO: benchmark against the version where P is part of the type is below
struct MaternP{T} <: IsotropicKernel{T}
    p::Int
    MaternP{T}(p::Int) where T = 0 ≤ p ? new(p) : throw(DomainError("p = $p is negative"))
end

MaternP(p::Int = 0) = MaternP{Float64}(p)
MaternP(k::Matern) = MaternP(floor(Int, k.ν)) # project Matern to closest MaternP

function (k::MaternP)(τ::Number)
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
struct Cosine{T, V<:Union{T, AbstractVector{T}}} <: StationaryKernel{T}
    μ::V
end
const Cos = Cosine
# TODO: look up trig-identity -> low-rank
(k::Cosine)(τ) = cos(2π * dot(k.μ, τ))
(k::Cosine{<:Number, <:Number})(τ) = cos(2π * k.μ * sum(τ))

parameters(k::Cosine) = k.μ isa Number ? [k.μ] : k.μ
nparameters(k::Cosine) = length(k.μ)

####################### spectral mixture kernel ################################
# can be seen as product kernel of Constant, Cosine, ExponentiatedQuadratic
Spectral(w::Real, μ, σ) = Product(Constant(w), Cos(μ), ARD(EQ(), 2π^2/σ.^2))
SpectralMixture(w::AbstractVector, μ, σ) = Sum(x->Spectral(x...), zip(w, μ, σ))
const SM = SpectralMixture

############################ Cauchy Kernel #####################################
# there is something else in the literature called with the same name ...
struct Cauchy{T} <: IsotropicKernel{T} end
Cauchy() = Cauchy{Float64}()
(k::Cauchy)(τ::Number) =  1/(π * (1+τ^2))

# for spectroscopy
PseudoVoigt(α::T) where T<:Number = α*EQ{T}() + (1-α)*Cauchy{T}()

###################### Inverse Multi-Quadratic  ################################
# seems formally similar to Cauchy, Cauchy is equal to power of IMQ
struct InverseMultiQuadratic{T} <: IsotropicKernel{T}
    c::T
end
(k::InverseMultiQuadratic)(τ::Number) = 1/√(τ^2 + k.c^2)

######################## Kernel Input Transfomations ###########################
############################ Length Scale ######################################
# TODO: using meta-programming, could write constructors:
# EQ(l::Float) = Lengthscale(EQ(), l)
# for all isotropic kernels?
struct Lengthscale{T, K} <: StationaryKernel{T}
    k::K
    l::T
    function Lengthscale(k::StationaryKernel, l::Real)
        if 0 > l; throw(DomainError("l = $l is non-positive")) end
        S = promote_type(eltype(k), typeof(l))
        l = convert(S, l)
        new{S, typeof(k)}(k, l)
    end
end
(k::Lengthscale)(τ::Number) = k.k(τ/k.l)
isisotropic(k::Lengthscale) = isisotropic(k.k)
isstationary(k::Lengthscale) = isstationary(k.k)

parameters(k::Lengthscale) = vcat(parameters(k.k), k.l)
nparameters(k::Lengthscale) = nparameters(k.k) + 1
function Base.similar(k::Lengthscale, θ::AbstractVector)
    n = checklength(k, θ)
    k = similar(k.k, @view(θ[1:n-1]))
    Lengthscale(k, θ[n])
end

########################### Change of Input Norm  ##############################
# apply a different norm to input radius r of a stationary kernel
# special case of input transformation functional
struct Normed{T, K<:IsotropicKernel{T}, N} <: StationaryKernel{T}
    k::K
    n::N # norm for r call
end

# TODO: could have a IsotropicNormed kernel if N <: IsotropicNorm
# currently, only relevant to Lᵖ norm
(m::Normed)(τ) = m.k(m.n(τ))
# WARNING: need to define parameters for norm function n (if it has any)
parameters(k::Normed) = vcat(parameters(k.k), parameters(k.n))
nparameters(k::Normed) = nparameters(k.k) + nparameters(k.n)
function Base.similar(k::Normed, θ::AbstractVector)
    checklength(k, θ)
    nk = nparameters(k.k)
    k = similar(k.k, @view(θ[1:nk]))
    n = similar(k.n, @view(θ[nk+1:end]))
    Normed(k, n)
end

# automatic relevance determination with length scale parameters l
function ARD(k::StationaryKernel, l::AbstractVector{<:Real})
    Normed(k, Metrics.EnergeticNorm(Diagonal(1 ./ l)))
end
function Energetic(k::StationaryKernel, A::AbstractMatOrFac{<:Real})
    Normed(k, Metrics.EnergeticNorm(A))
end
############################ periodic kernel ###################################
# derived by David MacKay
# input has to be 1D stationary or isotropic
struct Periodic{T, K<:StationaryKernel{T}} <: IsotropicKernel{T}
    k::K
end
# squared euclidean distance of x, y in the space (cos(x), sin(x))
# since (cos(x) - cos(y))^2 + (sin(x) - sin(y))^2 = 4*sin((x-y)/2)^2
(k::Periodic)(τ::Number) = k.k(sin(π*τ)^2)
parameters(k::Periodic) = parameters(k.k)
nparameters(k::Periodic) = nparameters(k.k)

########################### WIP / not relevant #################################
#### special Matern version
# struct MaternP{T, P} <: IsotropicKernel{T}
#     MaternP{T}(p::Int) where T = 0 ≤ p ? new{T, p}(p) : error("p is negative")
# end
#
# MaternP(p::Int = 0) = MaternP{Float64}(p)
# MaternP(k::Matern) = MaternP{eltype{k}}(floor(Int, k.ν)) # project Matern to closest MaternP
#
# function (k::MaternP{T, P})(τ::Real) where {T, P}
#     val = zero(τ)
#     r = sqrt(2P+1) * abs(τ)
#     for i in 0:P
#         val += (factorial(P+i)/(factorial(i)*factorial(P-i))) * (2r)^(P-i)
#     end
#     val *= exp(-r) * (factorial(P)/factorial(2P))
# end

# struct SM{T, V, U} <: StationaryKernel{T}
#     w::T # weight of Gaussian in frequency space
#     μ::V # mean of Gaussian in frequency space
#     σ::U # diagonal variance of
#     # SM(w, μ, σ) = (0 < w) && (0 .< σ) ? new(w, μ, σ) : error("error")
# end
# # distance could be vector of squared component distances
# function (k::SM)(τ::RTV)
#     w * cos(2π * dot(μ, τ)) * exp(-2π^2 * dot(τ.^2, σ^2))
# end

########################## Poly-harmonic spline ################################
# only conditionally p.s.d.
# # TODO: is T necessary here? might lead to problems, since promotion does not take place
# struct Polyharmonic{T, K} <: IsotropicKernel{T} end
# Polyharmonic(k::Integer) = Polyharmonic{Float64, k}()
# ThinPlate() = Polyharmonic{Float64, 2}()
#
# # PolyHarmonic{T}(k::Integer) where {T} = PolyHarmonicSpline{T, k}()
# # ThinPlate{T}() = PolyHarmonicSpline{T, 2}() # special case of PolyHarmonicSpline
#
# function (k::Polyharmonic{T, K})(τ::Number) where {T, K}
#     τ ≈ 0 ? zero(τ) : (iseven(K) ? τ^K * log(abs(τ)) : τ^K)
# end
