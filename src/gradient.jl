# implementation of derivative and gradient kernel
########################### Gradient kernel ####################################
abstract type AbstractDerivativeKernel{T, K} <: MultiKernel{T} end

# f ∼ GP(μ, k)
# ∂f ∼ GP(∂μ, dkxy) # gradient kernel
struct GradientKernel{T, K, IT<:InputTrait} <: AbstractDerivativeKernel{T, K}
    k::K
    input_trait::IT
    function GradientKernel{T, K}(k, it = input_trait(k)) where {T, K}
        new{T, K, typeof(it)}(k, it)
    end
end
GradientKernel(k::AbstractKernel{T}) where T = GradientKernel{T, typeof(k)}(k)
GradientKernel(k) = GradientKernel{Float64, typeof(k)}(k)
@inline input_trait(G::GradientKernel) = G.input_trait
# gramian_eltype(G::GradientKernel) = eltype(G.k)
Base.eltype(G::GradientKernel) = eltype(G.k)

function elsize(G::Gramian{<:AbstractMatrix, <:GradientKernel}, i::Int)
    i ≤ 2 ? length(G.x[1]) : 1
end

(G::GradientKernel)(x, y) = gradient_kernel(G.k, x, y, G.input_trait)

# O(d²) fallback for non-structured case
function gradient_kernel(k, x, y, ::GenericInput)
    K = allocate_gradient_kernel(k, x, y, GenericInput())
    gradient_kernel!(K, k, x, y, GenericInput())
end
function allocate_gradient_kernel(k, x, y, ::GenericInput)
    zeros(gramian_eltype(k, x, y), length(x), length(y))
end

function gradient_kernel!(K::AbstractMatrix, k, x, y, ::GenericInput)
    value = zeros(eltype(K), length(x)) # necessary?
    derivs = K # jacobian (higher order terms)
    result = DiffResults.DiffResult(value, K)
    g(y) = ForwardDiff.gradient(z->k(z, y), x)
    ForwardDiff.jacobian!(result, g, y)
    return K
end

# specialized lazy representation of element of gradient kernel gramian
# increases performance dramatically, compared to generic Woodbury implementation
struct GradientKernelElement{T, K, X, Y, IT} <: Factorization{T}
    k::K
    x::X
    y::Y
    input_trait::IT
end

function GradientKernelElement(k, x, y, IT::InputTrait = input_trait(k))
    T = gramian_eltype(k, x, y)
    GradientKernelElement{T, typeof(k), typeof(x), typeof(y), typeof(IT)}(k, x, y, IT)
end
Base.size(K::GradientKernelElement) = (length(K.x), length(K.y))
Base.size(K::GradientKernelElement, i::Int) = 1 ≤ i ≤ 2 ? size(K)[i] : 1
Base.eltype(K::GradientKernelElement{T}) where T = T
# gradient kernel element only used for sparsely representable elements
Base.Matrix(K::GradientKernelElement) = K * I(size(K, 1))

function GradientKernelElement{T}(k, x, y, it::InputTrait) where T
    GradientKernelElement{T, typeof(k), typeof(x), typeof(y), typeof(it)}(k, x, y, it)
end

function gradient_kernel(k, x, y, it::InputTrait)
    T = gramian_eltype(k, x, y)
    GradientKernelElement{T}(k, x, y, it)
end

function gradient_kernel!(K::GradientKernelElement, k, x, y, it::InputTrait)
    GradientKernelElement{eltype(K)}(k, x, y, it)
end

function Base.:*(G::GradientKernelElement, a)
    T = promote_type(eltype(G), eltype(a))
    b  = zeros(T, size(a))
    mul!(b, G, a)
end

const GenericGradientKernelElement{T, K, X, Y} = GradientKernelElement{T, K, X, Y, <:GenericInput}
const IsotropicGradientKernelElement{T, K, X, Y} = GradientKernelElement{T, K, X, Y, IsotropicInput}

# isotropic kernel
function LinearAlgebra.mul!(b, G::IsotropicGradientKernelElement, a, α::Number = 1, β::Number = 0) #, ::IsotropicInput = G.input_trait)
    r = difference(G.x, G.y) # other implementation appears to be faster for d = 128 since r does not have to be computed twice?
    r² = sum(abs2, r)
    k1, k2 = derivative_laplacian(G.k, r²) # IDEA: these could be computed once and stored in G
    dot_r_a = r'a # dot(r, a)
    @. b = α * -2(k1 * a + 2*k2 * r * dot_r_a) + β * b
end

# sparse but not completely lazy representation
function WoodburyFactorizations.Woodbury(K::IsotropicGradientKernelElement)
    k, x, y = K.k, K.x, K.y
    r = x - y
    r = reshape(r, :, 1) # do these work without reshape?
    r² = sum(abs2, r)
    d = length(x)
    k1, k2 = derivative_laplacian(k, r²)
    D = (-2k1*I)(d)
    C = MMatrix{1, 1}(-4k2)
    return K = Woodbury(D, r, C, r')
end

const DotProductGradientKernelElement{T, K, X, Y} = GradientKernelElement{T, K, X, Y, DotProductInput}

function LinearAlgebra.mul!(b, K::DotProductGradientKernelElement, a, α::Number = 1, β::Number = 0)
    k, x, y = K.k, K.x, K.y
    d² = dot(x, y)
    k1, k2 = derivative_laplacian(k, d²)
    dot_x_a = x'a # dot(x, a)
    @. b = α * (k1 * a + k2 * y * dot_x_a) + β * b
end

# sparse but not completely lazy representation
function WoodburyFactorizations.Woodbury(K::DotProductGradientKernelElement)
    k, x, y = K.k, K.x, K.y
    d² = dot(x, y)
    k1, k2 = derivative_laplacian(k, d²)
    D = (k1*I)(length(x))
    C = MMatrix{1, 1}(k2)
    return K = Woodbury(D, copy(y), C, copy(x)')
end

const LinearFunctionalGradientKernelElement{T, K, X, Y} = GradientKernelElement{T, K, X, Y, StationaryLinearFunctionalInput}

function LinearAlgebra.mul!(b, K::LinearFunctionalGradientKernelElement, a, α::Number = 1, β::Number = 0)
    k, x, y = K.k, K.x, K.y
    r = difference(x, y)
    cr = dot(k.c, r)
    k1, k2 = derivative_laplacian(k, cr)
    dot_c_a = k.c'a # dot(c, a)
    @. b = α * -k2 * k.c * dot_c_a + β * b
end

# Base.Matrix(K::LinearFunctionalGradientKernelElement) = Matrix(LazyMatrixProduct(K)) # or more generally: K * I(size(K, 1))
# sparse but not completely lazy representation
function LazyMatrixProduct(K::LinearFunctionalGradientKernelElement)
    k, x, y = K.k, K.x, K.y
    r = difference(x, y)
    cr = dot(k.c, r)
    k1, k2 = derivative_laplacian(k, cr)
    c = zeros(eltype(k.c), length(x), 1)
    @. c = k.c
    c2 = -k2*c
    return LazyMatrixProduct(c, c2')
end

function evaluate_block!(Gij, k::GradientKernel, x, y, IT = input_trait(k))
    gradient_kernel!(Gij, k.k, x, y, IT)
end

############################# Constant Kernel ##################################
# efficient specialization for constants
# IDEA: have special constant input trait, since it can combine with any kernel
gradient_kernel(k::Constant, x, y, ::IsotropicInput) = gradient_kernel(k, x, y)
gradient_kernel!(K, k::Constant, x, y, ::IsotropicInput) = gradient_kernel!(K, k, x, y)
function gradient_kernel(k::Constant, x, y)
    Zeros(length(x), length(y))
end
gradient_kernel!(K::Zeros, k::Constant, x, y) = K
gradient_kernel!(K::AbstractMatrix, k::Constant, x, y) = (K .= 0)

function gramian(k::GradientKernel{<:Any, <:Constant}, x::AbstractVector, y::AbstractVector)
    d = length(x[1])
    Zeros(length(x) * d, length(y) * d)
end
########################### Neural Network Kernel ##############################
# specialization of gradient nn kernel, written before more general AD-implementation,
# which obviates the need for special derivations like these
function allocate_gradient_kernel(k::NeuralNetwork, x, y, ::GenericInput)
    x isa SVector && (x = MVector(x))
    y isa SVector && (y = MVector(y))
    U = hcat(x, y) # this preserves static arrays
    D = Diagonal(zero(x))
    # C = similar(x, 2, 2)
    elty = typeof(k(x, y))
    C = zero(MMatrix{2, 2, elty})
    K = Woodbury(D, U, C, U')
end

# specialization for neural network kernel
# NOTE: assumes K is allocated with allocate_gradient_kernel
# IDEA: take care of σ constant of NN kernel (variance of bias term)
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
# might not be necessary anymore, benchmark against GradientKernel
struct ValueGradientKernel{T, K, IT<:InputTrait} <: AbstractDerivativeKernel{T, K}
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
function (G::ValueGradientKernel)(x, y)
    value_gradient_kernel(G.k, x, y, input_trait(G.k))
end

function value_gradient_kernel(k, x, y, IT::InputTrait = input_trait(G.k))
    d = length(x)
    kxy = k(x, y)
    value_value = kxy
    value_gradient = zeros(typeof(kxy), d)
    gradient_value = zeros(typeof(kxy), d)
    gradient_gradient = gradient_kernel(k, x, y, IT)
    K = DerivativeKernelElement(value_value, value_gradient, gradient_value, gradient_gradient)
    value_gradient_kernel!(K, k, x, y, IT)
end

# IDEA: for Sum and Product kernels, if input_trait is not passed, could default to Generic
# No, should keep track of input type in kernel
function value_gradient_kernel!(K::DerivativeKernelElement, k, x, y)
    value_gradient_kernel!(K, k, x, y, input_trait(k))
end
function value_gradient_kernel!(K::DerivativeKernelElement, k, x, y, IT::InputTrait)
    K.value_value = k(x, y)
    value_gradient_covariance!(K.gradient_value, K.value_gradient, k, x, y, IT)
    K.gradient_gradient = gradient_kernel!(K.gradient_gradient, k, x, y, IT)
    return K
end

# NOTE: since value_gradient_covariance! was added, specializations of value_gradient_kernel!
# only yield marginal performance improvements (<2x)
# could be removed to reduce LOC
function value_gradient_kernel!(K::DerivativeKernelElement, k, x, y, T::DotProductInput)
    xy = dot(x, y)
    k0, k1 = value_derivative(k, xy)
    K.value_value = k0
    @. K.value_gradient = k1*x # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.gradient_value = k1*y # ForwardDiff.gradient(z->k(z, y), x)
    K.gradient_gradient = gradient_kernel!(K.gradient_gradient, k, x, y, T)
    return K
end

function value_gradient_kernel!(K::DerivativeKernelElement, k, x, y, T::IsotropicInput)
    r = K.gradient_value
    @. r = x - y
    d² = sum(abs2, r)
    k0, k1 = value_derivative(k, d²)
    K.value_value = k0
    @. K.value_gradient = r
    @. K.value_gradient *= -2k1 # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.gradient_value *= 2k1 # ForwardDiff.gradient(z->k(z, y), x)
    K.gradient_gradient = gradient_kernel!(K.gradient_gradient, k, x, y, T)
    return K
end

# need this to work efficiently with BlockFactorizations, see blockmul! in gramian
# reuses temporary storage in Gij, speeds up MVM by more than an order of magnitude
function evaluate_block!(Gij::DerivativeKernelElement, k::ValueGradientKernel, x, y, IT::InputTrait = input_trait(k))
    value_gradient_kernel!(Gij, k.k, x, y, IT)
end

################################################################################
# computes covariance between value and gradient, needed for value-gradient kernel
# technically, complexity would be O(d) even with generic implementation
# in practice, at least an order of magitude can be gained by specialized implementations
function value_gradient_covariance!(gx, gy, k, x, y, ::GenericInput)
    # GradientConfig() # for generic version, this could be pre-computed for efficiency gains
    ForwardDiff.gradient!(gx, z->k(z, y), x) # these are bottlenecks
    ForwardDiff.gradient!(gy, z->k(x, z), y)
    return gx, gy
end

function value_gradient_covariance!(gx, gy, k, x, y, ::GenericInput, α::Real, β::Real)
    tx, ty = ForwardDiff.gradient(z->k(z, y), x), ForwardDiff.gradient(z->k(x, z), y)
    @. gx = α * tx + β * gx
    @. gy = α * ty + β * gy
    return gx, gy
end

function value_gradient_covariance!(gx, gy, k::Sum, x, y, ::GenericInput, α::Real = 1, β::Real = 0)
    @. gx *= β
    @. gy *= β
    for h in k.args
        value_gradient_covariance!(gx, gy, h, x, y, input_trait(h), α, 1)
    end
    return gx, gy
end

# this is more tricky to do without additional allocations
# could do it with one more vector
function value_gradient_covariance!(gx, gy, k::Product, x, y, ::GenericInput, α::Real = 1, β::Real = 0)
    @. gx *= β
    @. gy *= β
    prod_k_xy = k(x, y)
    if !iszero(prod_k_xy)
        for i in eachindex(k.args)
            ki = k.args[i]
            αi = α * prod_k_xy / ki(x, y)
            value_gradient_covariance!(gx, gy, ki, x, y, input_trait(ki), αi, 1)
        end
    else # GradientKernel of Product requires less efficient O(dr²) special case if prod_k_xy is zero, for now, throw error
        throw(DomainError("value_gradient_covariance! of Product not supported for products that are zero"))
    end
    return gx, gy
end

function value_gradient_covariance!(gx, gy, k, x, y, ::IsotropicInput, α::Real = 1, β::Real = 0)
    r = difference(x, y)
    r² = sum(abs2, r)
    k1 = ForwardDiff.derivative(k, r²) # IDEA: this computation could be pooled with the gradient computation
    @. gx = α * 2k1 * r + β * gx
    @. gy = -α * 2k1 * r + β * gy
    return gx, gy
end

function value_gradient_covariance!(gx, gy, k, x, y, ::DotProductInput, α::Real = 1, β::Real = 0)
    xy = dot(x, y)
    k1 = ForwardDiff.derivative(k, xy) # IDEA: this computation could be pooled with the gradient computation
    @. gx = α * k1 * y + β * gx
    @. gy = α * k1 * x + β * gy
    return gx, gy
end

function value_gradient_covariance!(gx, gy, k, x, y, ::StationaryLinearFunctionalInput, α::Real = 1, β::Real = 0)
    cr = dot(k.c, difference(x, y))
    k1 = ForwardDiff.derivative(k, cr) # IDEA: this computation could be pooled with the gradient computation
    @. gx = α * k1 * k.c + β * gx
    @. gy = -α * k1 * k.c + β * gy
    return gx, gy
end

################################################################################
# for 1d inputs
# can't be AbstractDerivativeKernel since it is not a multi-kernel
struct DerivativeKernel{T, K} <: AbstractKernel{T}
    k::K
end
DerivativeKernel(k) = DerivativeKernel{Float64, typeof(k)}(k)
function (G::DerivativeKernel)(x::Real, y::Real)
    g(y) = ForwardDiff.derivative(z->G.k(z, y), x)
    ForwardDiff.derivative(g, y)
end

################################################################################
struct ValueDerivativeKernel{T, K} <: AbstractDerivativeKernel{T, K}
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

############################# Helpers ##########################################
# computes value, first, and second derivatives of f at x
# this works even with nested differentiation
function value_derivative_laplacian(f, x::Real)
    fx = f(x)
    fx, derivative_laplacian(f, x, typeof(fx))...
end
# TODO: overwrite for MaternP
function derivative_laplacian(f, x::Real, T::DataType = typeof(x))
    g(x) = ForwardDiff.derivative(f, x)
    value_derivative(g, x, T)
end

# NOTE: if nested differentation through value_derivative is happening,
# need to pass correct output T = f(x)
function value_derivative(f, x::Real, T::DataType = typeof(x))
    r = DiffResults.DiffResult(zero(T), zero(T)) # this could take pre-allocated temporary storage
    r = ForwardDiff.derivative!(r, f, x)
    r.value, r.derivs[1]
end
