############################ stationary kernels ################################
# using LinearAlgebraExtensions: difference
# notation:
# x, y inputs
# τ = x-y difference of inputs
# r = norm(x-y) norm of τ
# r² = r^2
(k::StationaryKernel)(x, y) = k(difference(x, y)) # if the two argument signature is not defined, default to stationary
(k::IsotropicKernel)(x, y) = k(euclidean2(x, y)) # if the two argument signature is not defined, default to isotropic
(k::IsotropicKernel)(τ) = k(sum(abs2, τ)) # if only the scalar argument form is defined, must be isotropic

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

# should type of constant field and r agree? what promotion is necessary?
# do we need the isotropic/ stationary evaluation, if we overwrite the mercer one?
(k::Constant)() = k.c
(k::Constant)(r²) = k.c # stationary / isotropic
(k::Constant)(x, y) = k.c # mercer

gramian(k::Constant, x::AbstractVector, y::AbstractVector) = Fill(k.c, length(x), length(y))

#################### standard exponentiated quadratic kernel ###################
struct ExponentiatedQuadratic{T} <: IsotropicKernel{T} end
const EQ{T} = ExponentiatedQuadratic{T}
@functor EQ
EQ() = EQ{Union{}}() # defaults to "bottom" type since it doesn't have any parameters

(k::EQ)(r²::Number) = exp(-r² / 2)

########################## rational quadratic kernel ###########################
struct RationalQuadratic{T} <: IsotropicKernel{T}
    α::T # relative weighting of small and large length scales
    RationalQuadratic{T}(α) where T = (0 < α) ? new(α) : throw(DomainError("α not positive"))
end
const RQ = RationalQuadratic
@functor RQ
RQ(α::Real) = RQ{typeof(α)}(α)

(k::RQ)(r²::Number) = (1 + r² / (2*k.α))^-k.α

########################### exponential kernel #################################
struct Exponential{T} <: IsotropicKernel{T} end
const Exp = Exponential
Exp() = Exp{Union{}}()

(k::Exp)(r²::Number) = exp(-sqrt(r²))

############################ γ-exponential kernel ##############################
struct GammaExponential{T<:Real} <: IsotropicKernel{T}
    γ::T
    GammaExponential{T}(γ) where {T} = (0 ≤ γ ≤ 2) ? new(γ) : throw(DomainError("γ not in [0,2]"))
end
const γExp = GammaExponential
@functor γExp
γExp(γ::T) where T = γExp{T}(γ)

(k::γExp)(r²::Number) = exp(-r²^(k.γ/2) / 2)

########################### white noise kernel #################################
struct Delta{T} <: IsotropicKernel{T} end
@functor Delta
const δ = Delta
δ() = δ{Union{}}()

(k::δ)(r²) = all(iszero, r²) ? one(eltype(r²)) : zero(eltype(r²))
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
@functor Matern
Matern(ν::T) where {T} = Matern{T}(ν)

(k::Matern)(r²::Number) = Matern(r², k.ν)

# NOTE: this implementation only guarantees second order differentiability
# for higher orders, need to get higher order taylor expansion
@inline function Matern(r²::Number, ν::Real)
    ε = eps(typeof(r²))
    taylor_bound = (2 < ν) ? ε^(1/2) : ((1 < ν) ? ε : zero(ε))
    if r² < taylor_bound
        y = one(promote_type(typeof(r²), typeof(ν)))
        if ν > 1
            y += ν / (2*(1-ν)) * r² # first order
        end
        if ν > 2
            y += ν^2 / (8*(2 - 3ν + ν^2)) * (r²)^2 # second order
        end
        return y
    else
        r = sqrt(2ν*r²)
        2^(1-ν) / gamma(ν) * adbesselkxv(ν, r) # is equal to r^ν * besselk(ν, r)
    end
end

################# Matern kernel with ν = p + 1/2 where p ∈ ℕ ###################
struct MaternP{T, DT, CT} <: IsotropicKernel{T}
    p::Int
    derivatives::DT
    coefficients::CT
end

function MaternP(p::Int)
    0 > p && throw(DomainError("p = $p is negative"))
    d = MaternP_derivatives_at_zero(p)
    d = float(d)
    c = MaternP_coefficients(p)
    c = float(c)
    MaternP{Union{}, typeof(d), typeof(c)}(p, d, c)
end
MaternP(k::Matern) = MaternP(floor(Int, k.ν)) # project Matern to closest MaternP

(k::MaternP)(r²::Integer) = k(float(r²))
function (k::MaternP)(r²::Number)
    p = k.p
    r²_constant = (r² isa Taylor1 ? r².coeffs[1] : r²)
    taylor_bound = eps(typeof(r²_constant))^(1/p)
    use_taylor = r²_constant < taylor_bound
    if use_taylor # taylor_bound # around zero, use taylor expansion in r² to avoid singularity in derivative
        y = one(r²)
        r²i = r²
        for i in 1:p # iterating over r²^i allows for automatic differentiation (even though for a regular float, it would all be zero)
            y += k.derivatives[i] * r²i / factorial(i)
            r²i *= r²
        end
        return y
    else
        y = zero(r²)
        r = sqrt((2p+1)*r²)
        ri = one(r)
        for i in 1:p
            y += k.coefficients[i] * ri # (2r)^(p-i)
            ri *= 2r
        end
        y += ri # coefficient of (2r)^p is 1
        y *= exp(-r) / (factorial(2p) ÷ factorial(p))
    end
end

# naïve implementation, does not allow for differentiability at zero
# and recomputes coefficients based on factorials
@inline function MaternP(r²::Number, p::Int)
    val = zero(r²)
    r = sqrt((2p+1)*r²)
    for i in 0:p # IDEA: could table factorials and generate powers by iteration
        val += (factorial(p+i) ÷ (factorial(p-i) * factorial(i))) * (2r)^(p-i)
    end
    val *= exp(-r) / (factorial(2p) ÷ factorial(p))
end

# computes derivatives of k(r²) w.r.t. r² at zero
function MaternP_derivatives_at_zero(p::Int)
    r² = symbols(:r²) # uses SymEngine
    kr = MaternP(r², p)
    d = zeros(Rational{Int64}, p)
    for i in 1:p
        dkr = expand(diff(kr, r²))
        d[i] = dkr(0)
        kr = dkr
    end
    return d
end

function MaternP_coefficients(p::Int)
    coefficients = zeros(Rational{typeof(p)}, p)
    factorial_p = factorial(p)
    for i in 1:p
        coefficients[i] = (binomial(p, i) * (factorial(p+i) ÷ factorial_p))
    end
    return reverse(coefficients)
end

########################### cosine kernel ######################################
# interesting because it allows negative co-variances
# it is a valid stationary kernel,
# because it is the inverse Fourier transform of point measure at μ (delta distribution)
struct CosineKernel{T, V<:Union{T, AbstractVector{T}}} <: StationaryKernel{T}
    c::V
end
@functor CosineKernel
const Cosine = CosineKernel
const Cos = Cosine

# IDEA: trig-identity -> low-rank gramian
# NOTE: this is the only stationary non-isotropic kernel so far
input_trait(::CosineKernel) = StationaryLinearFunctionalInput() # dependent on dot(c, τ)
(k::CosineKernel)(c_dot_τ::Real) = cos(2π * c_dot_τ)
function (k::CosineKernel)(x, y)
    τ = difference(x, y)
    k(dot(k.c, τ))
end

####################### spectral mixture kernel ################################
# can be seen as product kernel of Constant, Cosine, ExponentiatedQuadratic
Spectral(w::Real, μ, l) = prod((w, Cosine(μ), ARD(EQ(), l))) # 2π^2/σ.^2)
SpectralMixture(w::AbstractVector, μ, l) = sum(Spectral.(w, μ, l))
const SM = SpectralMixture

############################ Cauchy Kernel #####################################
# there is something else in the literature with the same name ...
struct Cauchy{T} <: IsotropicKernel{T} end
@functor Cauchy
Cauchy() = Cauchy{Union{}}()
(k::Cauchy)(r²::Number) = inv(1+r²) # π is not necessary, we are not normalizing

# for spectroscopy
PseudoVoigt(α) = α*EQ() + (1-α)*Cauchy()

###################### Inverse Multi-Quadratic  ################################
# seems formally similar to Cauchy, Cauchy is equal to power of IMQ
struct InverseMultiQuadratic{T} <: IsotropicKernel{T}
    c::T
end
@functor InverseMultiQuadratic
(k::InverseMultiQuadratic)(r²::Number) = 1/√(r² + k.c^2)
