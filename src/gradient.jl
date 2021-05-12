# implementation of derivative and gradient kernel
import LazyLinearAlgebra: evaluate_block!

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
function evaluate_block!(K::AbstractMatOrFac, G::GradientKernel, x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G))
    gradient_kernel!(K, G.k, x, y, T)
end

function gradient_kernel(k, x::AbstractVector, y::AbstractVector, T::InputTrait) #::GenericInput = GenericInput())
    K = allocate_gradient_kernel(k, x, y, T)
    gradient_kernel!(K, k, x, y, T)
end

# allocates space for gradient kernel evaluation but does not evaluate
# separation from evaluation useful for ValueGradientKernel
allocate_gradient_kernel(k, x, y, ::GenericInput) = zeros(length(x), length(x))

function gradient_kernel!(K::AbstractMatrix, k, x::AbstractVector, y::AbstractVector, ::GenericInput)
    value = similar(x)
    derivs = K # jacobian (higher order terms)
    result = DiffResults.DiffResult(value, K)
    g(y) = ForwardDiff.gradient(z->k(z, y), x)
    ForwardDiff.jacobian!(result, g, y)
    return K
end

function allocate_gradient_kernel(k, x, y, T::DotProductInput)
    U = similar(y, length(y), 1)
    V = similar(x, length(x), 1)'
    A = Diagonal(similar(x))
    C = similar(x, 1, 1)
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
    r = similar(x, length(x), 1)
    D = Diagonal(similar(x))
    C = similar(x, 1, 1)
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
# specialization of gradient nn kernel, written before more general AD-implementation,
# which obviates the need for special derivations like these
function allocate_gradient_kernel(k::Constant, x, y, ::InputTrait)
    Zeros(length(x), length(y))
end
gradient_kernel!(K::Zeros, k::Constant, x, y, ::InputTrait) = K

########################### Neural Network Kernel ##############################
# specialization of gradient nn kernel, written before more general AD-implementation,
# which obviates the need for special derivations like these
function allocate_gradient_kernel(k::NeuralNetwork, x, y, ::GenericInput)
    U = hcat(x, y)
    D = Diagonal(similar(x))
    C = similar(x, 2, 2)
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

############################ Gradient Algebra ##################################
################################# Sum ##########################################
# allocates space for gradient kernel evaluation but does not evaluate
# separation from evaluation useful for ValueGradientKernel
function allocate_gradient_kernel(k::Sum, x, y, ::GenericInput)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    LazyMatrixSum(H...)
end

function gradient_kernel!(K::LazyMatrixSum, k::Sum, x::AbstractVector, y::AbstractVector, ::GenericInput)
    for (h, H) in zip(k.args, K.args)
        gradient_kernel!(H, h, x, y, input_trait(h))
    end
    return K
end

################################ Product #######################################
# for product kernel with generic input
function allocate_gradient_kernel(k::Product, x, y, ::GenericInput)
    d, r = length(x), length(k.args)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    T = typeof(k(x, y))
    A = LazyMatrixSum(
                (LazyMatrixProduct(Diagonal(zeros(T, d)), h) for h in H)...
                )
    U = zeros(T, (d, r))
    V = zeros(T, (d, r))
    C = Woodbury(I(r), ones(r), ones(r)', -1)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::Product, x::AbstractVector, y::AbstractVector, ::GenericInput = input_trait(k))
    A = W.A # this is a LazyMatrixSum of LazyMatrixProducts
    ForwardDiff.jacobian!(W.U, z->k(z, y), x)
    ForwardDiff.jacobian!(W.V', z->k(x, z), y)
    k_j = [h(x, y) for h in k.args]
    k_j .= prod(k_j) ./ k_j
    r = length(k.args)
    for i in 1:r
        h, H = k.args[i], A.args[i]
        D = H.args[1]
        @. D.diag = k_j
        gradient_kernel!(H.args[2], h, x, y, input_trait(h))
    end
    return W
end

############################# Separable Product ################################
# for product kernel with generic input
function allocate_gradient_kernel(k::SeparableProduct, x::AbstractVector{<:Number},
                                  y::AbstractVector{<:Number}, ::GenericInput)
    d = length(x)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    T = typeof(k(x, y))
    A = LazyMatrixProduct(Diagonal(zeros(T, d)), Diagonal(zeros(T, d)))
    U = Diagonal(zeros(T, d))
    V = Diagonal(zeros(T, d))
    C = Woodbury(I(r), ones(r), ones(r)', -1)
    Woodbury(A, U, C, V)
end

function gradient_kernel!(W::Woodbury, k::SeparableProduct, x::AbstractVector, y::AbstractVector, ::GenericInput = input_trait(k))
    A = W.A # this is a LazyMatrixProducts of Diagonals
    D, H = A.args # first is scaling matrix by leave_one_out_products, second is diagonal of derivative kernels
    for (i, ki) in enumerate(k.args)
        xi, yi = x[i], y[i]
        D.diag[i, i] = ki(xi, yi)
        W.U[i, i] = ForwardDiff.derivative(z->ki(z, yi), xi)
        W.V[i, i] = ForwardDiff.derivative(z->ki(xi, z), yi)
        H[i, i] = DerivativeKernel(ki)(xi, yi)
    end
    leave_one_out_products!(D.diag)
    return W
end

############################# Separable Sum ####################################
# IDEA: implement block separable with x::AbstractVecOfVec
function allocate_gradient_kernel(k::SeparableSum, x::AbstractVector{<:Number},
                                  y::AbstractVector{<:Number}, ::GenericInput)
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    D = Diagonal(d)
end

function gradient_kernel!(D::Diagonal, k::SeparableSum, x::AbstractVector{<:Number},
                          y::AbstractVector{<:Number}, ::GenericInput)
    for (i, ki) in enumerate(k.args)
        D[i, i] = DerivativeKernel(ki)(x[i], y[i])
    end
    return D
end

############################## Input Transformations ###########################
# can be most efficiently represented by factoring out the Jacobian w.r.t. input warping
function gramian(G::GradientKernel{<:Real, <:Warped},  x::AbstractVector, y::AbstractVector)
    W = G.k
    U(x) = BlockFactorization(Diagonal([ForwardDiff.jacobian(W.u, xi) for xi in x]))
    k = GradientKernel(W.k)
    LazyMatrixProduct(U(x)', gramian(k, x, y), U(y))
end

function gramian(G::GradientKernel{<:Real, <:ScaledInputKernel},  x::AbstractVector, y::AbstractVector)
    n, m = length(x), length(y)
    S = G.k
    Ux = kronecker(I(n), S.U)
    Uy = n == m ? Ux : kronecker(I(m), S.U)
    k = GradientKernel(S.k)
    LazyMatrixProduct(Ux', gramian(k, x, y), Uy)
end

function gramian(G::GradientKernel{<:Real, <:Lengthscale}, x::AbstractVector, y::AbstractVector)
    n, m = length(x), length(y)
    L = G.k
    Ux = Diagonal(fill(L.l, d*n)) # IDEA: Fill for lazy uniform array
    Uy = n == m ? Ux : Diagonal(fill(L.l, d*m))
    k = GradientKernel(L.k)
    LazyMatrixProduct(Ux', gramian(k, x, y), Uy)
end

############################### VerticalRescaling ##############################
# gradient element can be expressed with WoodburyIdentity and LazyMatrixProduct
function allocate_gradient_kernel(k::VerticalRescaling, x, y, ::GenericInput = GenericInput())
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    A = LazyMatrixProduct(Diagonal(fill(f(x), d)), H, Diagonal(fill(f(y), d)))
    U = zeros(d, 2)
    V = zeros(d, 2)
    C = zeros(2, 2)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::VerticalRescaling, x, y, ::GenericInput = GenericInput())
    f, h, A = k.f, k.k, W.A
    fx, fy = f(x), f(y)
    @. A.args[1].diag = fx
    H = A.args[2] # LazyMatrixProduct: first and third are the diagonal scaling matrices, second is the gradient_kernel_matrix of h
    @. A.args[3].diag = fy
    gradient_kernel!(H, h, x, y, input_trait(h))
    ForwardDiff.gradient!(@view(W.U[:, 1]), f, x)
    ForwardDiff.gradient!(@view(W.U[:, 2]), z->h(z, y), x)
    ForwardDiff.gradient!(@view(W.V[1, :]), f, y)
    ForwardDiff.gradient!(@view(W.V[2, :]), z->h(x, z), y)
    W.C[1, 1] = h(x, y)
    W.C[1, 2] = fy
    W.C[2, 1] = fx
    return W
end

############################ Scalar Chain Rule #################################
# generic implementation of scalar chain rule, does not require input kernel to have a basic input type
# gradient element can be expressed with WoodburyIdentity and LazyMatrixProduct
function allocate_gradient_kernel(k::Chained, x, y, ::GenericInput)
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    A = LazyMatrixProduct(Diagonal(fill(f(h(x, y)), d)), H)
    U = zeros(d, 1)
    V = zeros(d, 1)
    C = zeros(1, 1)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::Chained, x, y, ::GenericInput)
    f, h, A = k.f, k.k, W.A
    f1, f2 = derivative_laplacian(f, h(x, y))
    @. A.args[1].diag = f1
    H = A.args[2] # LazyMatrixProduct: first argument is diagonal scaling, second is the gradient_kernel_matrix of h
    gradient_kernel!(H, h, x, y, input_trait(h))
    ForwardDiff.gradient!(@view(W.U[:]), z->h(z, y), x)
    ForwardDiff.gradient!(@view(W.V[1, :]), z->h(x, z), y)
    @. W.C = f2
    return W
end

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
    A22 = allocate_gradient_kernel(k, x, y, T)
    A = if A22 isa Matrix
            [similar(x, 0, 0) for i in 1:2, j in 1:2]
        else
            Matrix{AbstractMatOrFac{eltype(x)}}(undef, 2, 2) # this could be more concrete if T = GenericInput
        end
    A[1, 1] = zeros(1, 1)
    A[1, 2] = zeros(1, d)
    A[2, 1] = zeros(d, 1)
    A[2, 2] = A22
    nindices = (1, 2, d+2)
    K = BlockFactorization(A, nindices, nindices)
    return value_gradient_kernel!(K, k, x, y, T)
end
# IDEA: specialize evaluate for IsotropicInput, DotProductInput
# returns block matrix
function value_gradient_kernel!(K::BlockFactorization, k, x::AbstractVector, y::AbstractVector, T::InputTrait)
    # value = zero(eltype(x))
    # derivs = K.A[1, 2]
    # r = DiffResults.DiffResult(value, derivs)
    # ForwardDiff.gradient!(r, z->k(x, z), y)
    # K.A[1, 1] .= r.value
    #
    # derivs = K.A[2, 1]
    # r = DiffResults.DiffResult(value, derivs)
    # ForwardDiff.gradient!(r, z->k(z, y), x)
    # println("here")
    # a = K.A[1, 1]
    # println(a)
    # println(typeof(a))
    # println(k(x, y))
    K.A[1, 1][1, 1] = k(x, y) # FIXME this line is problematic
    K.A[1, 2] .= ForwardDiff.gradient(z->k(x, z), y)' # ForwardDiff.gradient!(r, z->k(z, y), x)
    K.A[2, 1] .= ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.A[2, 2], k, x, y, T) # call GradientKernel for component
    return K
end

function value_gradient_kernel!(K::BlockFactorization, k, x::AbstractVector, y::AbstractVector, T::DotProductInput)
    f = _derivative_helper(k)
    xy = dot(x, y)
    k0, k1 = value_derivative(f, xy)
    K.A[1, 1][1, 1] = k0 # FIXME this line is problematic
    @. K.A[1, 2] = k1*x' # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.A[2, 1] = k1*y # ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.A[2, 2], k, x, y, T) # call GradientKernel for component
    return K
end

function value_gradient_kernel!(K::BlockFactorization, k, x::AbstractVector, y::AbstractVector, T::IsotropicInput)
    f = _derivative_helper(k)
    r = K.A[2, 1]
    @. r = x - y
    d² = sum(abs2, r)
    k0, k1 = value_derivative(f, d²)
    K.A[1, 1][1, 1] = k0 # FIXME this line is problematic
    @. K.A[1, 2] = r'
    @. K.A[1, 2] *= -2k1 # ForwardDiff.gradient(z->k(x, z), y)'
    @. K.A[2, 1] *= 2k1 # ForwardDiff.gradient(z->k(z, y), x)
    gradient_kernel!(K.A[2, 2], k, x, y, T) # call GradientKernel for component
    return K
end

function LazyLinearAlgebra.evaluate_block!(K::BlockFactorization, G::ValueGradientKernel, x::AbstractVector, y::AbstractVector, T::InputTrait = input_trait(G.k))
    value_gradient_kernel!(K, G.k, x, y, T)
end

# function evaluate!(K::AbstractMatrix, g::ValueGradientKernel, x::AbstractVector, y::AbstractVector, ::GenericInput)
#     function value_gradient(k, x::AbstractVector, y::AbstractVector)
#         r = DiffResults.GradientResult(y) # this could take pre-allocated temporary storage
#         ForwardDiff.gradient!(r, z->k(z, y), x)
#         r = vcat(r.value, r.derivs[1]) # d+1 # ... to avoid this
#     end
#     println(size(K))
#     value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
#     derivs = @view K[:, 2:end] # jacobian of helper (higher order terms)
#     result = DiffResults.DiffResult(value, derivs)
#     println(result)
#     ForwardDiff.jacobian!(result, z->value_gradient(g.k, x, z), y)
#     display(K)
#     return K
# end

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

# derivative helper returns the function f such that f(r²) = k(x, y) where r² = sum(abs2, x-y)
# this is required for the efficient computation of the gradient kernel matrices
_derivative_helper(k) = throw("_derivative_helper not defined for kernel of type $(typeof(k))")
_derivative_helper(k::EQ) = f(r²) = exp(-r²/2)
_derivative_helper(k::RQ) = f(r²) = inv(1 + r² / (2*k.α))^k.α
_derivative_helper(k::Constant) = f(r²) = k.c
_derivative_helper(k::Dot) = f(r²) = r²
_derivative_helper(k::ExponentialDot) = f(r²) = exp(r²)
_derivative_helper(k::Power) = f(r²) = _derivative_helper(k.k)(r²)^k.p
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
