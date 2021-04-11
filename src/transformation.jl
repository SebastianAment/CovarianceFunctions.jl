######################## Kernel Input Transfomations ###########################
############################ Length Scale ######################################
# TODO: using meta-programming, could write constructors:
# EQ(l::Float) = Lengthscale(EQ(), l)
# for all isotropic kernels?
struct Lengthscale{T, K} <: StationaryKernel{T}
    k::K
    l::T
    function Lengthscale(k::StationaryKernel, l::Real)
        0 > l && throw(DomainError("l = $l is non-positive"))
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
struct Normed{T, K, N} <: StationaryKernel{T}
    k::K
    n::N # norm for r call
    function Normed(k::AbstractKernel{T}, n) where T
        new{T, typeof(k), typeof(n)}(k, n)
    end
    function Normed(k, n)
        T = typeof(k(n(0.)))
        new{T, typeof(k), typeof(n)}(k, n)
    end
end
# WARNING: input function must be isotropic
(m::Normed)(τ::AbstractVector) = m.k(m.n(τ))
(m::Normed)(x, y) = m(difference(x, y))

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
function ARD(k, l::AbstractVector{<:Real})
    f(x) = enorm(Diagonal(inv.(l)), x)
    Normed(k, f)
end
ARD(k, l::Real) = Lengthscale(k, l)
function Energetic(k, A::AbstractMatOrFac{<:Real})
    f(x) = enorm(A, x)
    Normed(k, f)
end
############################ periodic kernel ###################################
# derived by David MacKay
# input has to be 1D stationary or isotropic
struct Periodic{T, K<:StationaryKernel{T}} <: IsotropicKernel{T}
    k::K
end
# squared euclidean distance of x, y in the space (cos(x), sin(x))
# since τ_new^2 = (cos(x) - cos(y))^2 + (sin(x) - sin(y))^2 = 4*sin((x-y)/2)^2 = 4sin(τ/2)^2
(p::Periodic)(τ::Number) = p.k(2sin(π*τ)) # without scaling, this is 1-periodic
parameters(p::Periodic) = parameters(p.k)
nparameters(p::Periodic) = nparameters(p.k)
Base.similar(p::Periodic, θ::AbstractVector) = similar(p.k, θ)

######################## general linear input scaling ##########################
struct ScaledInputKernel{T, K, UT} <: AbstractKernel{T}
    k::K
    U::UT
end
ScaledInputKernel(k, U) = ScaledInputKernel{Float64}(k, U)
ScaledInputKernel{T}(k, U) where T = ScaledInputKernel{T, typeof(k), typeof(U)}(k, U)
(S::ScaledInputKernel)(x, y) = S.k(S.U*x, S.U*y)

# if U has quadratic complexity in d, this reduces from O(n^2d^2) to O(nd^2 + n^2d)
# currently, this acceleration is currently only used if S.U is square and not diagonal
function gramian(S::ScaledInputKernel, x::AbstractVector, y::AbstractVector)
    if size(S.U, 1) < size(S.U, 2)
        Gramian(S, x, y)
    else
        x = (S.U,) .* x
        y = x ≡ y ? x : (S.U,) .* y
        gramian(S.k, x, y)
    end
end
# if the scaling can be carried out efficiently, like for Diagonal, no need
function gramian(S::ScaledInputKernel{<:Real, <:Any, <:Diagonal},  x::AbstractVector, y::AbstractVector)
    Gramian(S, x, y)
end
