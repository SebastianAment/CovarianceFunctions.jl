######################## Kernel Input Transfomations ###########################
############################ Length Scale ######################################
# IDEA: using meta-programming, could write constructors:
# EQ(l::Float) = Lengthscale(EQ(), l)
# for all isotropic kernels?
struct Lengthscale{T, K} <: IsotropicKernel{T}
    k::K
    l::T
    function Lengthscale(k::IsotropicKernel, l)
        all(>(0), l) || throw(DomainError("l = $l is non-positive"))
        length(l) == 1 || throw(DimensionMismatch("lengthscale l has to has length 1"))
        S = promote_type(eltype(k), typeof(l))
        l = convert(S, l)
        new{S, typeof(k)}(k, l)
    end
end
@functor Lengthscale
# IDEA: for optmization with Zygote, could let k.l be vector and write sum(r² ./ k.l)
(k::Lengthscale)(r²::Number) = k.k(r² / k.l[1]^2)

########################### Change of Input Norm  ##############################
# apply a different norm to input radius r of an isotropic kernel
# special case of input transformation functional
# NOTE: needs
struct Normed{T, K, N} <: StationaryKernel{T}
    k::K
    n²::N # norm for r call
    function Normed(k::AbstractKernel{T}, n²) where T
        new{T, typeof(k), typeof(n²)}(k, n²)
    end
    function Normed(k, n²)
        T = typeof(k(n²(0.)))
        new{T, typeof(k), typeof(n²)}(k, n²)
    end
end
@functor Normed
# WARNING: input function must be isotropic
(m::Normed)(τ) = m.k(m.n²(τ))
(m::Normed)(x, y) = m(difference(x, y))

# automatic relevance determination with length scale parameters l
function ARD(k, l::AbstractVector{<:Real})
    f(x) = enorm2(Diagonal(inv.(l)), x)
    Normed(k, f)
end
ARD(k, l::Real) = Lengthscale(k, l)
function Energetic(k, A::AbstractMatOrFac{<:Real})
    f(x) = enorm2(A, x)
    Normed(k, f)
end
############################ periodic kernel ###################################
# derived by David MacKay
# input has to be 1D stationary or isotropic
struct Periodic{T, K<:IsotropicKernel{T}} <: StationaryKernel{T}
    k::K
end
@functor Periodic
# squared euclidean distance of x, y in the space (cos(x), sin(x))
# since τ_new^2 = (cos(x) - cos(y))^2 + (sin(x) - sin(y))^2 = 4*sin((x-y)/2)^2 = 4sin(τ/2)^2
# (p::Periodic)(x, y) = p(difference(x, y))
function (p::Periodic)(τ::Number)
    r² = (2sin(π*τ))^2 # since isotropic kernels are called with r²
    p.k(r²) # without scaling, this is 1-periodic
end
parameters(p::Periodic) = parameters(p.k)
nparameters(p::Periodic) = nparameters(p.k)
Base.similar(p::Periodic, θ::AbstractVector) = similar(p.k, θ)

######################## Kernel Input Transfomations ###########################
######################## general linear input scaling ##########################
struct ScaledInputKernel{T, K, UT} <: AbstractKernel{T}
    k::K
    U::UT
end
@functor ScaledInputKernel
ScaledInputKernel(k, U) = ScaledInputKernel{Float64}(k, U)
ScaledInputKernel{T}(k, U) where T = ScaledInputKernel{T, typeof(k), typeof(U)}(k, U)
(S::ScaledInputKernel)(x, y) = S.k(S.U*x, S.U*y)
(S::ScaledInputKernel)(r) = S.k(S.U*r) # for stationary kernels

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
# if the scaling can be carried out efficiently, like for Diagonal, no need to pre-compute scaled data
function gramian(S::ScaledInputKernel{<:Real, <:Any, <:Diagonal},  x::AbstractVector, y::AbstractVector)
    Gramian(S, x, y)
end

############################### input warping ##################################
struct Warped{T, K, U} <: AbstractKernel{T}
    k::K
    u::U
end
@functor Warped
Warped(k, u) = Warped{Float64}(k, u)
Warped{T}(k, u) where T = Warped{T, typeof(k), typeof(u)}(k, u)
function Warped{T}(k, U::AbstractMatOrFac) where T
    u(x) = U*x
    Warped{T, typeof(k), typeof(u)}(k, u)
end
(W::Warped)(x, y) = W.k(W.u(x), W.u(y))

# if U has quadratic complexity in d, this reduces from O(n^2d^2) to O(nd^2 + n^2d)
# currently, this acceleration is currently only used if S.U is square and not diagonal
function gramian(W::Warped, x::AbstractVector, y::AbstractVector)
    x = W.u.(x)
    y = x ≡ y ? x : W.u.(y)
    gramian(W.k, x, y)
end
# if the scaling can be carried out efficiently, like for Diagonal, no need to pre-compute scaled data
function gramian(W::Warped{<:Real, <:Any, <:Diagonal}, x::AbstractVector, y::AbstractVector)
    Gramian(W, x, y)
end

############################ Symmetric Kernel ##################################
# make this useable for multi-dimensional inputs!
# in more dimensions, could have more general axis of symmetry
struct SymmetricKernel{T, K<:AbstractKernel} <: AbstractKernel{T}
    k::K # kernel to be symmetrized
    z::T # center
end
@functor SymmetricKernel
SymmetricKernel(k::AbstractKernel{T}) where T = SymmetricKernel(k, zero(T))

# for 1D axis symmetry
function (k::SymmetricKernel)(x, y)
    x -= k.z; y -= k.z;
    (k.k(x, y) + k.k(-x, y))/2
end

######################## Kernel Output Transformations #########################
############################ Scalar Chain Rule #################################
struct Chained{T, F, K} <: AbstractKernel{T}
    f::F
    k::K # requires k to be a kernel function
end
@functor Chained
Chained(f, k::AbstractKernel{T}) where T = Chained{T}(f, k)
Base.:∘(f, k::AbstractKernel) = Chained(f, k)

(C::Chained)(x, y) = C.f(C.k(x, y))
input_trait(C::Chained) = input_trait(C.k)

############################## rescaled kernel #################################
# IDEA: should be called DiagonalRescaling in analogy to the matrix case
# diagonal rescaling of covariance functions
# generalizes multiplying by constant kernel to multiplying by function
struct VerticalRescaling{T, K<:AbstractKernel{T}, F} <: AbstractKernel{T}
    k::K
    f::F
end
@functor VerticalRescaling

(k::VerticalRescaling)(x, y) = k.f(x) * k.k(x, y) * k.f(y)
# preserve structure if k.k is stationary and x, y are regular grids,
# since that introduces Toeplitz structure
function gramian(k::VerticalRescaling, x::AbstractVector, y::AbstractVector)
    # Diagonal(k.a.(x)) * gramian(k.k, x, y) * Diagonal(k.a.(y)))
    Dx = Diagonal(k.f.(x))
    Dy = Diagonal(k.f.(y))
    K = gramian(k.k, x, y)
    return LazyMatrixProduct(Dx, K, Dy)
end

# normalizes an arbitary kernel so that k(x,x) = 1
normalize(k::AbstractKernel) = VerticalRescaling(k, x->1/√k(x, x))
