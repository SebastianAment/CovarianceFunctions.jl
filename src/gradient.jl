# implementation of derivative and gradient kernel
########################### Gradient kernel ####################################
# f ∼ GP(μ, k)
# ∂f ∼ GP(∂μ, dkxy)
# [f, ∂f] ∼ GP([μ, ∂μ], dK)
# IDEA: have efficient implementation for stationary kernel
struct GradientKernel{T, K} <: MultiKernel{T}
    k::K
    # could add temporary storage for gradient calculation here
end
const Gradient = GradientKernel
const Derivative = GradientKernel
GradientKernel(k::AbstractKernel{T}) where {T} = GradientKernel{T, typeof(k)}(k)

function elsize(G::Gramian{<:AbstractMatrix, <:GradientKernel}, i::Int)
    return i ≤ 2 ? length(G.x[1]) + 1 : 1
end

# (Darian) Derivative Kernel: This function handles building the dxd matrix
# (assuming scalar inputs) of the covariances between the kernel associated
# with the function and its derivative.
function (k::GradientKernel)(x::AbstractVector, y::AbstractVector)
    n = checklength(x, y) + 1
    K = zeros(n, n)
    return evaluate!(K, k, x, y)
end

function evaluate!(K::AbstractMatrix, g::GradientKernel, x::AbstractVector, y::AbstractVector)
    function value_gradient(k, x::AbstractVector, y::AbstractVector)
        r = DiffResults.GradientResult(y) # this could take pre-allocated temporary storage
        FD.gradient!(r, z->k(z, y), x)
        vcat(r.value, r.derivs[1]) # d+1 # ... to avoid this
    end
    value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
    derivs = @view K[:, 2:end] # jacobian of helper (higher order terms)
    result = DiffResults.DiffResult(value, derivs)
    ForwardDiff.jacobian!(result, z->value_gradient(g.k, x, z), y)
    return K
end

# 1d-input
function (k::GradientKernel)(x::Real, y::Real)
    evaluate!(zeros(typeof(x), (2, 2)), k, x, y)
end

function evaluate!(K::AbstractMatrix, g::GradientKernel, x::Real, y::Real)
    function value_derivative(k, x::Real, y::Real)
        r = DiffResults.DiffResult(zero(y), zero(y)) # this could take pre-allocated temporary storage
        r = FD.derivative!(r, z->k(z, y), x)
        vcat(r.value, r.derivs[1])
    end
    value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
    derivs = @view K[:, 2] # jacobian of helper (higher order terms)
    result = DiffResults.DiffResult(value, derivs)
    ForwardDiff.jacobian!(result, z->value_derivative(g.k, x, z[1]), [y])
    return K
end
