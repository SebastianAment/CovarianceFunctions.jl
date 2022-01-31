# IDEA: GradientLaplacianKernel?
struct HessianKernel{T, K} <: MultiKernel{T}
    k::K
    # input_type::IT  IT<:InputTrait
    # could A temporary storage for gradient calculation here
end
input_trait(G::HessianKernel) = input_trait(G.k)
HessianKernel(k::AbstractKernel{T}) where {T} = HessianKernel{T, typeof(k)}(k)
HessianKernel(k) = HessianKernel{Float64, typeof(k)}(k) # use fieldtype here?

function elsize(G::Gramian{<:AbstractMatrix, <:HessianKernel}, i::Int)
    i ≤ 2 ? length(G.x[1])^2 : 1
end

# IDEA: input_trait(G.k) could be pre-computed
function (G::HessianKernel)(x::AbstractVector, y::AbstractVector)
    hessian_kernel(G.k, x, y, input_trait(G.k))
end

function hessian_kernel(k, x::AbstractVector, y::AbstractVector, T::InputTrait)#::GenericInput = GenericInput())
    K = allocate_hessian_kernel(k, x, y, T)
    hessian_kernel!(K, k, x, y, T)
end

########################## Generic Hessian Kernel ##############################
# allocates space for gradient kernel evaluation but does not evaluate
# separation from evaluation useful for ValueHessianKernel
function allocate_hessian_kernel(k, x, y, ::InputTrait)
    d = length(x)
    zeros(d^2, d^2)
end

function hessian_kernel!(K::AbstractMatrix, k, x::AbstractVector, y::AbstractVector, ::InputTrait)
    d = length(x)
    value = similar(x, d^2)
    derivs = K # Hessian of Hessian (higher order terms)
    result = DiffResults.DiffResult(value, derivs)
    g(y) = vec(ForwardDiff.hessian(z->k(z, y), x)) # covariance between vectorized Hessian and value
    ForwardDiff.jacobian!(result, z->ForwardDiff.jacobian(g, z), y) # hessian of vector valued function
    return K
end

############################ Lazy Kernel Element ###############################
# struct HessianKernelElement{T, K, X<:AbstractVector{T}, Y} <: Factorization{T}
#     k::K
#     x::X
#     y::Y
#     # input_trait::IT
# end
#
# # HessianKernelElement(k, x, y) = HessianKernelElement(k, x, y, input_trait(k))
# function Base.size(K::HessianKernelElement)
#     d = length(K.x)
#     (d^2, d^2)
# end
# Base.size(K::HessianKernelElement, i::Int) = 0 < i ≤ 2 ? size(K)[i] : 1

# function allocate_hessian_kernel(k, x, y, T::Union{IsotropicInput, DotProductInput})
#     HessianKernelElement(k, copy(x), copy(y))
# end

# function hessian_kernel!(K::HessianKernelElement, k, x::AbstractVector, y::AbstractVector, T::Union{IsotropicInput, DotProductInput})
#     K.x .= x; K.y .= y
#     return K
# end

# function BlockFactorizations.evaluate_block!(K::HessianKernelElement, G::HessianKernel, x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G))
#     hessian_kernel!(K, G.k, x, y, T)
# end

########################## Isotropic Hessian Kernel ############################
struct IsotropicHessianKernelElement{T, K, R<:AbstractVector{T}, RR<:AbstractMatrix,
                                     UT, CT, KD} <: Factorization{T}
    k::K
    r::R
    rr::RR # r*r'
    U::UT # for [vId, rr]
    C::CT
    kd::KD

    # these could have different element type depending on input
    # Ua::
    # CUa::
    # Ar::AT
    # BA
end
function IsotropicHessianKernelElement(k, x, y)
    # universal variables
    d = length(x)
    Id = I(d)
    vId = vec(Id)

    r = x - y
    rr = r * r'
    vrr = vec(rr)
    T = eltype(r)
    U = hcat(vId, vrr) # micro optimization IDEA: could link second column to rr via view
    C = zeros(T, (2, 2))

    r² = sum(abs2, r)
    f = _derivative_helper(k)
    kd = derivatives(f, r², 4)[2:end] # get first to fourth derivative
    @. kd = kd * 2^(1:4) # constant adjustment since kernels are function of r², not r²/2

    # Ua = zeros(T, 2)
    # CUa = zeros(T, 2)
    # Ar = zeros(T, d)
    # BA = zeros(T, (d, d))
    IsotropicHessianKernelElement(k, r, rr, U, C, kd) #, Ua, CUa, Ar, BA)
end
function Base.size(K::IsotropicHessianKernelElement)
    d = length(K.r)
    (d^2, d^2)
end
Base.size(K::IsotropicHessianKernelElement, i::Int) = 0 < i ≤ 2 ? size(K)[i] : 1

function allocate_hessian_kernel(k, x, y, ::IsotropicInput)
    IsotropicHessianKernelElement(k, copy(x), copy(y))
end

function BlockFactorizations.evaluate_block!(K::IsotropicHessianKernelElement,
    G::HessianKernel, x::AbstractVector, y::AbstractVector, ::IsotropicInput = IsotropicInput())
    hessian_kernel!(K, G.k, x, y)
end

function hessian_kernel!(K::IsotropicHessianKernelElement, k, x::AbstractVector, y::AbstractVector, ::IsotropicInput = IsotropicInput())
    @. K.r = x-y
    @. K.rr = K.r*K.r'
    r² = sum(abs2, K.r)
    f = _derivative_helper(k)
    K.kd .= derivatives(f, r², 4)[2:end] # get first to fourth derivative
    @. K.kd = K.kd * 2^(1:4) # constant adjustment since kernels are function of r², not r²/2

    @. K.U[:, 2] = $vec(K.rr) # [vec(Id) vec(rr)], only rr changes
    K.C[1, 1] = K.kd[2]
    K.C[2, 1] = K.kd[3]
    K.C[1, 2] = K.kd[3]
    K.C[2, 2] = K.kd[4]
    return K
end

function Base.Matrix(K::IsotropicHessianKernelElement)
    r = K.r
    rr = vec(K.rr)
    r² = sum(abs2, r)

    d = length(r)
    S = perfect_shuffle(d)

    B = kron(I(d), r*r') + kron(r*r', I(d))
    HH = K.U * K.C * K.U' # Hessian-Hessian
    HH .+= (S + I(d^2)) * (K.kd[2] * I(d^2) + K.kd[3] * B)
end

function LinearAlgebra.mul!(b::AbstractVector, K::IsotropicHessianKernelElement,
                            a::AbstractVector, α::Real = 1, β::Real = 0)
    d = length(K.r)
    d^2 == length(a) == length(b) || throw(DimensionMismatch())
    # below: multiplying by
    # [vec(Id) vec(rr)] * [kd[2] kd[3]; kd[3] kd[4]] * [vec(Id) rr]' i.e. U * C * U'
    # mul!(K.Ua, U', a) # Ua = U'a # this is tricky, because a might have different type
    # mul!(K.CUa, C, K.Ua) # CUa = C*U'a
    Ua = K.U' * a # more general, and these are 2 vectors
    CUa = K.C * Ua
    mul!(b, K.U, CUa, α, β)

    # below: multiplying with (Id2 + Shuffle)
    kd = K.kd
    A = reshape(a, d, d)
    @. b += α * kd[2] * a # I*a
    at = vec(A') # S*a where S is perfect_shuffle
    @. b += α * kd[2] * at

    # below: multiplying with
    # (Id2 + Shuffle) * f3(rn) * B, where
    # B = kronecker(Id, r*r') + kronecker(r*r', Id)
    # BA = zeros(d, d)
    # Ar = zeros(d)
    # mul!(K.Ar, A, K.r)
    # @. K.BA += (K.Ar*K.r') # A * rr' # this could in principle act on b
    # mul!(K.Ar, A', K.r)
    # @. K.BA += (K.r*K.Ar') # rr * A
    # @. b += α * kd[3] * ($vec(K.BA) + $vec(K.BA'))

    Ar = A * K.r # d vector
    BA = Ar * K.r' # A * rr' - d x d matrix
    mul!(Ar, A', K.r)
    @. BA += K.r * Ar' # rr * A
    @. b += α * kd[3] * $vec(BA)
    @. b += α * kd[3] * $vec(BA')
    return b
end

####################### Hessian DotProductInput Kernel #########################
struct DotProductHessianKernelElement{T, K, X<:AbstractVector{T}, Y<:AbstractVector,
                XX<:AbstractMatrix{T}, YY<:AbstractMatrix, KD} <: Factorization{T}
    k::K
    x::X
    y::Y
    xx::XX
    yy::YY
    kd::KD
end
function DotProductHessianKernelElement(k, x, y)
    # universal variables
    d = length(x)
    Id, Id2 = I(d), I(d^2)
    vId = vec(Id)
    xy = dot(x, y)
    xx = x*x'
    yy = y*y'
    f = _derivative_helper(k)
    kd = derivatives(f, xy, 4)[2:end]
    DotProductHessianKernelElement(k, x, y, xx, yy, kd)
end
function Base.size(K::DotProductHessianKernelElement)
    d = length(K.x)
    (d^2, d^2)
end
Base.size(K::DotProductHessianKernelElement, i::Int) = 0 < i ≤ 2 ? size(K)[i] : 1

function allocate_hessian_kernel(k, x, y, ::DotProductInput)
    DotProductHessianKernelElement(k, copy(x), copy(y))
end
# necessary for BlockFactorization
function BlockFactorizations.evaluate_block!(K::DotProductHessianKernelElement,
    G::HessianKernel, x::AbstractVector, y::AbstractVector, ::DotProductInput = DotProductInput())
    hessian_kernel!(K, G.k, x, y)
end
function hessian_kernel!(K::DotProductHessianKernelElement, k, x::AbstractVector, y::AbstractVector, ::DotProductInput = DotProductInput())
    @. K.x = x
    @. K.y = y
    @. K.xx = x*x'
    @. K.yy = y*y'
    xy = dot(x, y)
    f = _derivative_helper(K.k)
    K.kd .= derivatives(f, xy, 4)[2:end] # get first to fourth derivative
    return K
end
function LinearAlgebra.mul!(b::AbstractVector, K::DotProductHessianKernelElement, a::AbstractVector,
                            α::Real, β::Real)
    d = size(K.xx, 1)
    d^2 == length(a) == length(b) || throw(DimensionMismatch())
    xx, yy = vec(K.xx), vec(K.yy)

    # below: multiplying by yy * xx'
    @. b *= β
    xxa = dot(xx, a)
    @. b += α * K.kd[4] * yy * xxa

    # below: multiplying with (Id2 + Shuffle)
    A = reshape(a, d, d)
    @. b += α * K.kd[2] * a # I*a
    @. b += α * K.kd[2] * $vec(A') # + S*a where S is perfect_shuffle

    # below: multiplying with kron(I(d), yx) + kron(yx, I(d))
    BA = zeros(d, d)
    Av = zeros(d)
    mul!(Av, A, K.x)
    @. BA += (Av*K.y') # A * y*x' # this could in principle act on b
    mul!(Av, A', K.x)
    @. BA += (K.y*Av') # y*x' * A
    @. b += α * K.kd[3] * $vec(BA)
    @. b += α * K.kd[3] * $vec(BA')
    return b
end
function Base.Matrix(K::DotProductHessianKernelElement)
    d = length(K.x)
    # universal variables
    Id2 = I(d^2)
    S = perfect_shuffle(d)
    # specific variables
    yx = K.y*K.x'
    xx = vec(K.xx)
    yy = vec(K.yy)
    kd = K.kd # get first to fourth derivative
    B = kron(I(d), yx) + kron(yx, I(d))
    HH = (Id2 + S) * (kd[2] * Id2 + kd[3] * B) + kd[4] * (yy * xx')
end

################################################################################
# kernel for value, gradient, and Hessian observations
struct ValueGradientHessianKernel{T, K} <: MultiKernel{T}
    k::K
    # input_type::IT  IT<:InputTrait
end
input_trait(G::ValueGradientHessianKernel) = input_trait(G.k)
ValueGradientHessianKernel(k::AbstractKernel{T}) where {T} = ValueGradientHessianKernel{T, typeof(k)}(k)
ValueGradientHessianKernel(k) = ValueGradientHessianKernel{Float64, typeof(k)}(k) # use fieldtype here?

# IDEA: input_trait(G.k) could be pre-computed
function (G::ValueGradientHessianKernel)(x::AbstractVector, y::AbstractVector)
    value_gradient_hessian_kernel(G.k, x, y, input_trait(G.k))
end
function value_gradient_hessian_kernel(k, x, y, T::InputTrait)
    K = allocate_value_gradient_hessian_kernel(k, x, y, T)
    value_gradient_hessian_kernel!(K, k, x, y, T)
end
# allocating dense element (this should probably never be done in practice, only comparison)
function allocate_value_gradient_hessian_kernel(k, x, y, ::InputTrait)
    d = length(x)
    n = d^2 + d + 1
    zeros(n, n)
end
function value_gradient_hessian_kernel!(K::AbstractMatrix, k, x::AbstractVector, y::AbstractVector, ::InputTrait)
    d = length(x)
    # block indices
    vi = 1 # value index
    gi = (1+1):(d+1) # gradient index
    hi = (d+1+1):(1+d+d^2) # hessian index
    # value-value
    K[1, 1] = k(x, y)
    # value-gradient
    K[1, gi] = ForwardDiff.gradient(z->k(x, z), y)'
    K[gi, 1] = ForwardDiff.gradient(z->k(z, y), x)
    # gradient-gradient
    K[gi, gi] = ForwardDiff.jacobian(z2->ForwardDiff.gradient(z1->k(z1, z2), x), y)
    # value-hessian
    K[1, hi] = vec(ForwardDiff.hessian(z->k(x, z), y))'
    K[hi, 1] = vec(ForwardDiff.hessian(z->k(z, y), x))
    # gradient-hessian
    K[gi, hi] = ForwardDiff.jacobian(z1->vec(ForwardDiff.hessian(z2->k(z1, z2), y)), x)'
    K[hi, gi] = ForwardDiff.jacobian(z2->vec(ForwardDiff.hessian(z1->k(z1, z2), x)), y)
    # hessian-hessian
    g(y) = vec(ForwardDiff.hessian(z->k(z, y), x)) # covariance between vectorized Hessians
    H = ForwardDiff.jacobian(z->ForwardDiff.jacobian(g, z), y)
    K[hi, hi] .= reshape(H, d^2, d^2)
    return K
end

# this is neccesary when evaluate!(::ValueGradientHessianKernel) is called
# defines mul!
# TODO: increase pre-allocation
# TODO: unify this with DerivativeKernelElement
struct ValueGradientHessianKernelElement{T, K, X<:AbstractVector{T}, Y} <: Factorization{T}
   k::K
   x::X
   y::Y
   # r
   # U::UT # for [vId, rr]
   # Ua
   # CUa
   # C::CT
   # Ar::AT
   # BA
end

function value_gradient_hessian_kernel!(K::ValueGradientHessianKernelElement, k, x::AbstractVector, y::AbstractVector, T::Union{IsotropicInput, DotProductInput})
    K.x .= x; K.y .= y
    return K
end

function BlockFactorizations.evaluate_block!(K::ValueGradientHessianKernelElement,
    G::ValueGradientHessianKernel, x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G.k))
    value_gradient_hessian_kernel!(K, G.k, x, y, T)
end

function Base.size(K::ValueGradientHessianKernelElement)
    d = length(K.x)
    n = d^2 + d + 1
    (n, n)
end
Base.size(K::ValueGradientHessianKernelElement, i::Int) = 0 < i ≤ 2 ? size(K)[i] : 1

function Base.Matrix(K::ValueGradientHessianKernelElement) #, ::DotProductInput = input_trait(K.k))
    k, x, y = K.k, K.x, K.y
    d = length(x)
    n = d^2 + d + 1
    A = zeros(n, n)
    value_gradient_hessian_kernel!(A, k, x, y, input_trait(k))
    # d = length(x)
    # # universal variables
    # Id2 = I(d^2)
    # S = perfect_shuffle(d)
    #
    # # specific variables
    # xy = dot(x, y)
    # yx = y*x'
    # xx = vec(x*x')
    # yy = vec(y*y')
    #
    # f = _derivative_helper(k)
    # kd = derivatives(f, xy, 4)[2:end] # get first to fourth derivative
    #
    # B = kron(I(d), yx) + kron(yx, I(d))
    # HH = (Id2 + S) * (kd[2] * Id2 + kd[3] * B) +  kd[4] * (yy * xx')
end
function allocate_value_gradient_hessian_kernel(k, x, y, ::IsotropicInput)
    ValueGradientHessianKernelElement(k, copy(x), copy(y))
end

function LinearAlgebra.mul!(b::AbstractVector, K::ValueGradientHessianKernelElement,
                            a::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(b, K, a, α, β, input_trait(K.k))
end
# fast multiplication with isotropic second-order derivative kernel
function LinearAlgebra.mul!(b::AbstractVector, K::ValueGradientHessianKernelElement,
                            a::AbstractVector, α::Real, β::Real, T::IsotropicInput)
    k, x, y = K.k, K.x, K.y
    d = length(x)
    # block indices
    vi = 1 # value index
    gi = 2:(d+1) # gradient index
    hi = (d+2):(1+d+d^2) # hessian index

    avi, agi, ahi = @views a[vi], a[gi], a[hi]

    # universal variables
    Id, Id2 = I(d), I(d^2)
    vId = vec(Id)

    # specific variables
    r = x - y
    rr = vec(r*r')
    r² = norm(r)^2

    # evaluate kernel and derivatives
    k = K.k
    f = _derivative_helper(k)
    kd = derivatives(f, r², 4)
    @. kd *= 2^(0:4)
    k0, k1, k2, k3, k4 = kd

    @. b *= β # scaling of target

    # important inner products
    ra = dot(r, agi)
    ida = dot(vId, ahi)
    rra = dot(rr, ahi)

    # value
    b[vi] += α * k0 * a[vi] # value
    b[vi] -= α * k1 * ra # gradient
    b[vi] += α * (k1 * ida + k2 * rra) # hessian

    # gradient
    @. b[gi] += α * k1 * r * a[vi]
    @. b[gi] -= α * (k1 * agi + k2 * r * ra)
    @. b[gi] += α * (k2 * ida + k3 * rra) * r # this sign seems correct

    BA = zeros(d)
    Ahi = reshape(ahi, d, d)
    BA += Ahi * r + vec(r' * Ahi) # can be combined with the below
    @. b[gi] += α * k2 * BA

    # hessian
    @. b[hi] += α * (k1 * vId + k2 * rr) * a[vi] # value part of gradient
    @. b[hi] -= α * (k2 * vId + k3 * rr) * ra # this can be collected with the above
    BA = zeros(d, d)
    @. BA += agi * r' + r * agi' # can be combined with the below
    @. b[hi] -= α * k2 * $vec(BA)

    # hessian-hessian
    # below: multiplying by
    # [vec(Id) rr] * [kd[2] kd[3]; kd[3] kd[4]] * [vec(Id) rr]'
    U = hcat(vId, rr)
    C = zeros(2, 2)
    C[1, 1] = k2
    C[2, 1] = k3
    C[1, 2] = k3
    C[2, 2] = k4
    # Ua = U'a
    Ua = [ida, rra] # (U'*ahi)
    bhi = @view b[hi]
    mul!(bhi, U, C*Ua, α, 1)

    # below: multiplying with (Id2 + Shuffle) * ahi
    Ahi = reshape(ahi, d, d)
    @. b[hi] += α * k2 * ahi # I*a
    @. b[hi] += α * k2 * $vec(Ahi') # + S*a where S is perfect_shuffle

    # below: multiplying with
    # (Id2 + Shuffle) * f3(rn) * (Bi - Bj), where
    # B = kronecker(Id, r*r') + kronecker(r*r', Id)
    BA = zeros(d, d)
    Ar = zeros(d)
    mul!(Ar, Ahi, r)
    @. BA += (Ar*r') # A * rr' # this could in principle act on b
    mul!(Ar, Ahi', r)
    @. BA += (r*Ar') # rr * A

    @. b[hi] += α * k3 * $vec(BA)
    @. b[hi] += α * k3 * $vec(BA') # Adding Shuffle * A
    return b
end

# # multiplying with rank 1 kronecker sum
# # function _kron_sum_mul!(b::AbstractVector, x::AbstractVector, y::Abstractor,
# #                         a::AbstractVector, α::Real = 1, β::Real = 0)
# #
# # end
#


# block implementation probably less efficient than the Element one
# TODO: move to value_gradient_hessian_kernel!
# function value_gradient_hessian_kernel(K::BlockFactorization, k, x::AbstractVector, y::AbstractVector, T::DotProductInput)
#
#     # first, call ValueGradientKernel to calculate value-gradient covariances in upper 2x2 block of K
#     value_gradient_kernel!(K, k, x, y, T)
#
#     # universal variables
#     Id, Id2 = I(d), I(d^2)
#     vId = vec(Id)
#     S = perfect_shuffle(d)
#
#     # # specific variables
#     # r = x - y
#     # rr = vec(r*r')
#     # r² = norm(r)^2
#
#     f = _derivative_helper(k)
#     kd = derivatives(f, r², 4)[2:end] # get first to fourth derivative
#     kd = @. kd * 2^(1:4) # constant adjustment
#
#     # value-hessian
#     K[1, 3] = kd[1]
#     K[3, 1] = -1
#
#     # gradient-hessian
#     K[2, 3] = -1
#     K[3, 2] = -1
#
#     # hessian-hessian
#     K[3, 3] = -1
# end
