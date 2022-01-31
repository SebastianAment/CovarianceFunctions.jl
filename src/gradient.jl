# implementation of derivative and gradient kernel
########################### Gradient kernel ####################################
# f ∼ GP(μ, k)
# ∂f ∼ GP(∂μ, dkxy) # gradient kernel
struct GradientKernel{T, K, IT<:InputTrait} <: MultiKernel{T}
    k::K
    input_trait::IT
    function GradientKernel{T, K}(k, it = input_trait(k)) where {T, K}
        new{T, K, typeof(it)}(k, it)
    end
end
GradientKernel(k::AbstractKernel{T}) where T = GradientKernel{T, typeof(k)}(k)
GradientKernel(k) = GradientKernel{Float64, typeof(k)}(k)
input_trait(G::GradientKernel) = G.input_trait

function elsize(G::Gramian{<:AbstractMatrix, <:GradientKernel}, i::Int)
    i ≤ 2 ? length(G.x[1]) : 1
end

function (G::GradientKernel)(x::AbstractVector, y::AbstractVector)
    gradient_kernel(G.k, x, y, input_trait(G.k))
end

# necessary for blockmul! of BlockFactorization
function BlockFactorizations.evaluate_block!(K::AbstractMatOrFac, G::GradientKernel,
            x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G))
    gradient_kernel!(K, G.k, x, y, T)
end

function gradient_kernel(k, x::AbstractVector, y::AbstractVector, T::InputTrait) #::GenericInput = GenericInput())
    K = allocate_gradient_kernel(k, x, y, T)
    gradient_kernel!(K, k, x, y, T)
end

# allocates space for gradient kernel evaluation but does not evaluate
# separation from evaluation useful for ValueGradientKernel
# x and y should have same length, i.e. dimensionality
allocate_gradient_kernel(k, x, y, ::GenericInput) = zeros(length(x), length(y))

function gradient_kernel!(K::AbstractMatrix, k, x::AbstractVector, y::AbstractVector, ::GenericInput)
    value = similar(x)
    derivs = K # jacobian (higher order terms)
    result = DiffResults.DiffResult(value, K)
    g(y) = ForwardDiff.gradient(z->k(z, y), x)
    ForwardDiff.jacobian!(result, g, y)
    return K
end

function allocate_gradient_kernel(k, x, y, T::DotProductInput)
    U = reshape(zero(y), :, 1)
    V = reshape(zero(x), :, 1)'
    A = Diagonal(zero(x))
    C = MMatrix{1, 1}(k(x, y))
    W = Woodbury(A, U, C, V) # needs to be separate from IsotropicInput,
end

# NOTE: assumes K is allocated with allocate_gradient_kernel
function gradient_kernel!(K::Woodbury, k, x::AbstractVector, y::AbstractVector, ::DotProductInput)
    f = _derivative_helper(k)
    d² = dot(x, y)
    k1, k2 = derivative_laplacian(f, d²)
    @. K.A.diag = k1
    @. K.U = y
    @. K.C = k2
    @. K.V = x'
    return K
end

function allocate_gradient_kernel(k, x, y, T::IsotropicInput)
    r = reshape(x - y, :, 1) # do these work without reshape?
    kxy = k(x, y)
    d = length(x)
    D = Diagonal(MVector{d, typeof(kxy)}(undef)) # not using Fill here, because value can't be changed
    C = MMatrix{1, 1}(kxy)
    K = Woodbury(D, r, C, r')
end

# NOTE: assumes K is allocated with allocate_gradient_kernel
function gradient_kernel!(K::Woodbury, k, x::AbstractVector, y::AbstractVector, ::IsotropicInput)
    f = _derivative_helper(k)
    r = K.U
    @. r = x - y
    d² = sum(abs2, r)
    k1, k2 = derivative_laplacian(f, d²)
    @. K.A.diag = -2k1
    K.C[1, 1] = -4k2
    return K
end

############################# Constant Kernel ##################################
# efficient specialization for constants
function allocate_gradient_kernel(k::Constant, x, y, ::InputTrait)
    Zeros(length(x), length(y))
end
gradient_kernel!(K::Zeros, k::Constant, x, y, ::InputTrait) = K

########################### Neural Network Kernel ##############################
# specialization of gradient nn kernel, written before more general AD-implementation,
# which obviates the need for special derivations like these
function allocate_gradient_kernel(k::NeuralNetwork, x, y, ::GenericInput)
    U = hcat(x, y) # this preserves static arrays
    D = Diagonal(zero(x))
    # C = similar(x, 2, 2)
    elty = typeof(k(x, y))
    C = zero(MMatrix{2, 2, elty})
    K = Woodbury(D, U, C, U')
end

# specialization for neural network kernel
# NOTE: assumes K is allocated with allocate_gradient_kernel
# TODO: take care of σ constant of NN kernel (variance of bias term)
function gradient_kernel!(K::Woodbury, k::NeuralNetwork, x, y, ::GenericInput = GenericInput())
    Nx, Ny = dot(x, x) + 1, dot(y, y) + 1
    dxy, Nxy = dot(x, y), Nx * Ny
    d = dxy / sqrt(Nxy)
    f0(x) = 2/π * asin(x)
    f1(x) = 2/π * inv(sqrt(1-x^2)) # first derivative
    f2(x) = 2/π * inv(sqrt(1-x^2))^3 * x # second derivative
    k0, k1, k2 = f0(d), f1(d), f2(d) # kernel value and derivatives
    k1 /= sqrt(Nxy)
    k2 /= Nxy # incorporate normalization to simplify following expression

    # gradient
    # gradx = k1 / sqrt(Nxy) * (y - dxy / Nx * x) # this appears correct

    # second order derivatives
    @. K.A.diag = k1
    @. K.U[:, 1] = x
    @. K.U[:, 2] = y
    k1k2dxy = (k1 + k2 * dxy)
    K.C[1, 1] = - k1k2dxy / Nx # after normalization, dxy could become d
    K.C[2, 1] = k2
    K.C[1, 2] = k1k2dxy * dxy / Nxy
    K.C[2, 2] = - k1k2dxy / Ny
    return K
end

################################################################################
# structure that holds the potentially heterogeneous types of covariances between
# a kernel value and its derivatives
# mutable only because of value_value?
mutable struct DerivativeKernelElement{T, VV, VG, GV, GG, VH, HV, GH, HG, HH, GI, HI} <: AbstractMatrix{T}
    value_value::VV # value-value covariance (IDEA: if this is MMAtrix, could make structure non-mutable)
    value_gradient::VG # value-gradient covariance
    gradient_value::GV # gradient-value covariance
    gradient_gradient::GG # ...
    value_hessian::VH
    hessian_value::HV
    gradient_hessian::GH
    hessian_gradient::HG
    hessian_hessian::HH
    gradient_indices::GI
    hessian_indices::HI
end

function DerivativeKernelElement(vv, vg, gv, gg, vh = nothing, hv = nothing, gh = nothing, hg = nothing, hh = nothing)
    t = (vv, vg, gv, gg, vh, hv, gh, hg, hh)
    s = something(t...) # get the first input that's not nothing
    T = eltype(s)
    gi = gradient_indices(vv, gg)
    hi = hessian_indices(vv, gg, hh)
    DerivativeKernelElement{T, typeof.(t)..., typeof(gi), typeof(hi)}(t..., gi, hi)
end

function gradient_indices(value_value, gradient_gradient)
    i = 0
    if !isnothing(value_value)
        i += 1
    end
    if !isnothing(gradient_gradient)
        d = size(gradient_gradient, 1)
        return (i + 1) : (i + d)
    else
        return nothing
    end
end

function hessian_indices(value_value, gradient_gradient, hessian_hessian)
    i = 0
    if !isnothing(value_value)
        i += 1
    end
    if !isnothing(gradient_gradient)
        d = size(gradient_gradient, 1)
        i += d
    end
    if !isnothing(hessian_hessian)
        dd = size(hessian_hessian, 1)
        return (i + 1) : (i + dd)
    else
        return nothing
    end
end

Base.eltype(A::DerivativeKernelElement{T}) where {T} = T
function Base.size(A::DerivativeKernelElement)
    n = 0
    if !isnothing(A.value_value)
        n += 1
    end
    if !isnothing(A.gradient_gradient)
        n += size(A.gradient_gradient, 1)
    end
    if !isnothing(A.hessian_hessian)
        n += size(A.hessian_hessian, 1)
    end
    return (n, n)
end
Base.size(A::DerivativeKernelElement, i::Int) = i <= 2 ? size(A)[i] : 1

# TODO: efficient indexing
function Base.getindex(A::DerivativeKernelElement, i::Int, j::Int)
    # Matrix(A)[i, j]
    @boundscheck begin
        n = size(A, 1)
        (1 <= i <= n && 1 <= j <= n) || throw(BoundsError("attempt to access $(size(A)) DerivativeKernelElement at index ($i, $j)"))
    end
    gi = A.gradient_indices
    hi = A.hessian_indices
    if !isnothing(A.value_value)
        (i == 1 && j == 1) && return A.value_value
        if i == 1
            (!isnothing(gi) && j in gi) && return A.value_gradient[findfirst(==(j), gi)]
            (!isnothing(hi) && j in hi) && return A.value_hessian[findfirst(==(j), hi)]
        end
        if j == 1
            (!isnothing(gi) && i in gi) && return A.gradient_value[findfirst(==(i), gi)]
            (!isnothing(hi) && i in hi) && return A.hessian_value[findfirst(==(i), hi)]
        end
    end
    if !isnothing(gi)
        (i in gi && j in gi) && return A.gradient_gradient[findfirst(==(i), gi), findfirst(==(j), gi)]
        if !isnothing(hi)
            (i in gi && j in hi) && return A.gradient_hessian[findfirst(==(i), gi), findfirst(==(j), hi)]
            (i in hi && j in gi) && return A.hessian_gradient[findfirst(==(i), hi), findfirst(==(j), gi)]
        end
    end
    if !isnothing(hi)
        return A.hessian_hessian[findfirst(==(i), hi), findfirst(==(j), hi)]
    end
    error("this should never happen")
end

function LinearAlgebra.mul!(y::AbstractVector, A::DerivativeKernelElement, x::AbstractVector, α::Number = 1, β::Number = 0)
    y .*= β
    gi = A.gradient_indices
    hi = A.hessian_indices
    if !isnothing(A.value_value)
        y[1] += α * A.value_value * x[1]
    end
    if !isnothing(A.value_gradient)
        xg, yg = @views x[gi], y[gi]
        y[1] += α * dot(A.value_gradient, xg)
        @. yg += α * A.gradient_value * x[1]
    end
    if !isnothing(A.gradient_gradient)
        xg, yg = @views x[gi], y[gi]
        mul!(yg, A.gradient_gradient, xg, α, 1)
    end
    if !isnothing(A.value_hessian)
        xh, yh = @views x[hi], y[hi]
        y[1] += α * dot(A.value_hessian, xh)
        @. yh += α * A.hessian_value * x[1]
    end
    if !isnothing(A.gradient_hessian)
        xg, yg = @views x[gi], y[gi]
        xh, yh = @views x[hi], y[hi]
        mul!(yg, A.gradient_hessian, xh, α, 1)
        mul!(yh, A.hessian_gradient, xg, α, 1)
    end
    if !isnothing(A.hessian_hessian)
        xh, yh = @views x[hi], y[hi]
        mul!(yh, A.hessian_hessian, xh, α, 1)
    end
    return y
end

function Base.Matrix(A::DerivativeKernelElement)
    n = size(A, 1)
    M = zeros(eltype(A), n, n)
    gradient_index = 0
    hessian_index = 0
    if !isnothing(A.value_value)
        M[1, 1] = A.value_value
        gradient_index += 1 # if we have a value variance, gradient indices are 2:2+d
        hessian_index += 1
    end
    if !isnothing(A.value_gradient)
        d = length(A.value_gradient)
        @. M[2:1+d, 1] = A.gradient_value
        @. M[1, 2:1+d] = A.value_gradient
    end
    if !isnothing(A.gradient_gradient)
        d = size(A.gradient_gradient, 1)
        gi = gradient_index+1:gradient_index+d
        M[gi, gi] .= Matrix(A.gradient_gradient)
        hessian_index += d + 1 # if we have a gradient covariances, hessian indices are (1 + hasvalue? + d + 1)
    end
    if !isnothing(A.value_hessian)
        dd = length(A.value_hessian)
        hi = hessian_index+1:hessian_index+dd
        @. M[1, hi] = A.value_hessian
        @. M[hi, 1] = A.hessian_value
    end
    if !isnothing(A.gradient_hessian)
        d, dd = size(A.gradient_hessian)
        gi = gradient_index+1:gradient_index+d
        hi = hessian_index+1:hessian_index+dd
        M[gi, hi] .= Matrix(A.gradient_hessian)
        M[hi, gi] .= Matrix(A.hessian_gradient)
    end
    if !isnothing(A.hessian_hessian)
        dd = size(A.hessian_hessian, 1)
        hi = hessian_index+1:hessian_index+dd
        M[hi, hi] .= Matrix(A.hessian_hessian)
    end
    return M
end

########################## using new kernel element ############################
################################################################################
# [f, ∂f] ∼ GP([μ, ∂μ], dK) # value + gradient kernel
# IDEA: For efficiency, maybe create ValueGradientKernelElement like in hessian.jl
# currently, this is an order of magnitude slower than GradientKernel
struct ValueGradientKernel{T, K, IT<:InputTrait} <: MultiKernel{T}
    k::K
    input_trait::IT
    function ValueGradientKernel{T, K}(k, it = input_trait(k)) where {T, K}
        new{T, K, typeof(it)}(k, it)
    end
end
ValueGradientKernel(k::AbstractKernel{T}) where {T} = ValueGradientKernel{T, typeof(k)}(k)
ValueGradientKernel(k) = ValueGradientKernel{Float64, typeof(k)}(k) # use fieldtype here?

input_trait(G::ValueGradientKernel) = G.input_trait

function elsize(G::Gramian{<:AbstractMatrix, <:ValueGradientKernel}, i::Int)
    i ≤ 2 ? length(G.x[1]) + 1 : 1
end

# computes covariances of the function and its derivative.
function (G::ValueGradientKernel)(x::AbstractVector, y::AbstractVector)
    value_gradient_kernel(G.k, x, y, input_trait(G.k))
end

function value_gradient_kernel(k, x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G.k))
    d = length(x)
    kxy = k(x, y)
    value_value = kxy
    value_gradient = MVector{d, typeof(kxy)}(undef)
    gradient_value = MVector{d, typeof(kxy)}(undef)
    gradient_gradient = allocate_gradient_kernel(k, x, y, T)
    K = DerivativeKernelElement(value_value, value_gradient, gradient_value, gradient_gradient)
    value_gradient_kernel!(K, k, x, y, T)
end

# IDEA: specialize evaluate for IsotropicInput, DotProductInput
# returns block matrix
function value_gradient_kernel!(K::DerivativeKernelElement, k, x::AbstractVector, y::AbstractVector, T::InputTrait)
    K.value_value = k(x, y)
    K.value_gradient .= ForwardDiff.gradient(z->k(x, z), y) # ForwardDiff.gradient!(r, z->k(z, y), x)
    K.gradient_value .= ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.gradient_gradient, k, x, y, T) # call GradientKernel for component
    return K
end

function value_gradient_kernel!(K::DerivativeKernelElement, k, x::AbstractVector, y::AbstractVector, T::DotProductInput)
    f = _derivative_helper(k)
    xy = dot(x, y)
    k0, k1 = value_derivative(f, xy)
    K.value_value = k0
    @. K.value_gradient = k1*x # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.gradient_value = k1*y # ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.gradient_gradient, k, x, y, T) # call GradientKernel for component
    return K
end

function value_gradient_kernel!(K::DerivativeKernelElement, k, x::AbstractVector, y::AbstractVector, T::IsotropicInput)
    f = _derivative_helper(k)
    r = K.gradient_value
    @. r = x - y
    d² = sum(abs2, r)
    k0, k1 = value_derivative(f, d²)
    K.value_value = k0
    @. K.value_gradient = r
    @. K.value_gradient *= -2k1 # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.gradient_value *= 2k1 # ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.gradient_gradient, k, x, y, T) # call GradientKernel for component
    return K
end

function BlockFactorizations.evaluate_block!(K::DerivativeKernelElement, G::ValueGradientKernel,
            x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G.k))
    value_gradient_kernel!(K, G.k, x, y, T)
end

################################################################################
# for 1d inputs
struct DerivativeKernel{T, K} <: AbstractKernel{T}
    k::K
end
DerivativeKernel(k) = DerivativeKernel{Float64, typeof(k)}(k)
function (G::DerivativeKernel)(x::Real, y::Real)
    g(y) = ForwardDiff.derivative(z->G.k(z, y), x)
    ForwardDiff.derivative(g, y)
end

################################################################################
struct ValueDerivativeKernel{T, K} <: MultiKernel{T}
    k::K
end
ValueDerivativeKernel(k) = ValueDerivativeKernel{Float64, typeof(k)}(k)
function (k::ValueDerivativeKernel)(x::Real, y::Real)
    x = float(x)
    x, y = promote(x, y)
    value_derivative_kernel!(zeros(eltype(x), (2, 2)), k, x, y)
end
function value_derivative_kernel!(K::AbstractMatrix, g::ValueDerivativeKernel, x::Real, y::Real)
    function f(k, x::Real, y::Real)
        r = DiffResults.DiffResult(zero(y), zero(y)) # this could take pre-allocated temporary storage
        r = ForwardDiff.derivative!(r, z->k(z, y), x)
        vcat(r.value, r.derivs[1])
    end
    value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
    derivs = @view K[:, 2] # jacobian of helper (higher order terms)
    result = DiffResults.DiffResult(value, derivs)
    ForwardDiff.jacobian!(result, z->f(g.k, x, z[1]), [y])
    return K
end

# 1d-input, still necessary?
# computes 2x2 covariance matrix associated with the function and its derivative
# function (G::ValueGradientKernel)(x::Real, y::Real)
#     x = float(x)
#     x, y = promote(x, y)
#     value_gradient_kernel!(zeros(eltype(x), (2, 2)), G, x, y)
# end
#
# function value_gradient_kernel(G::ValueGradientKernel, x::Real, y::Real)
#     value_gradient_kernel!(zeros(2, 2), G.k, x, y)
# end
#
# function value_gradient_kernel!(K::AbstractMatrix, g::GradientKernel, x::Real, y::Real)
#     function value_derivative(k, x::Real, y::Real)
#         r = DiffResults.DiffResult(zero(y), zero(y)) # this could take pre-allocated temporary storage
#         r = ForwardDiff.derivative!(r, z->k(z, y), x)
#         vcat(r.value, r.derivs[1])
#     end
#     value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
#     derivs = @view K[:, 2] # jacobian of helper (higher order terms)
#     result = DiffResults.DiffResult(value, derivs)
#     ForwardDiff.jacobian!(result, z->value_derivative(g.k, x, z[1]), [y])
#     return K
# end

############################# Helpers ##########################################
# special cases that avoid problems with ForwardDiff.gradient and norm at 0
# IDEA: could define GradientKernel on euclidean2, dot and take advantage of chain rule
(k::EQ)(x, y) = exp(-euclidean2(x, y)/2)
(k::RQ)(x, y) = (1 + euclidean2(x, y) / (2*k.α))^-k.α
# (k::Lengthscale)(x, y) = k.k(euclidean2(x, y)/2)

# derivative helper returns the function f such that f(r²) = k(x, y) where r² = sum(abs2, x-y)
# this is required for the efficient computation of the gradient kernel matrices
_derivative_helper(k) = throw("_derivative_helper not defined for kernel of type $(typeof(k))")
_derivative_helper(k::EQ) = f(r²) = exp(-r²/2)
_derivative_helper(k::RQ) = f(r²) = inv(1 + r² / (2*k.α))^k.α
_derivative_helper(k::Constant) = f(r²) = k.c
_derivative_helper(k::Dot) = f(r²) = r² # r² = dot(x, y) in the case of dot product kernels
_derivative_helper(k::ExponentialDot) = f(r²) = exp(r²)
_derivative_helper(k::Power) = f(r²) = _derivative_helper(k.k)(r²)^k.p
function _derivative_helper(k::Lengthscale)
    f(r²) = _derivative_helper(k.k)(r²/k.l^2)
end

function _derivative_helper(k::Sum)
    summands = _derivative_helper.(k.args)
    f(r²) = sum(h->h(r²), summands)
end
function _derivative_helper(k::Product)
    factors = _derivative_helper.(k.args)
    f(r²) = prod(h->h(r²), factors)
end
# TODO: SeparableProduct, SeparableSum, what to do with vertical scaling?

# computes value, first, and second derivatives of f at x
function value_derivative_laplacian(f, x::Real)
    f(x), derivative_laplacian(f, x)...
end
# TODO: overwrite for MaternP
function derivative_laplacian(f, x::Real)
    g(x) = ForwardDiff.derivative(f, x)
    value_derivative(g, x)
end

# WARNING: do not use if nested differentation through value_derivative is happening
function value_derivative(f, x::Real)
    r = DiffResults.DiffResult(zero(x), zero(x)) # this could take pre-allocated temporary storage
    r = ForwardDiff.derivative!(r, f, x)
    r.value, r.derivs[1]
end
