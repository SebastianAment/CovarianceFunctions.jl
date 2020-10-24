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
GradientKernel(k::AbstractKernel{T}) where {T} = GradientKernel{T, typeof(k)}(k)

# compute the (i, j) entry of k(x, y)
Base.getindex(K::GradientKernel, i::Integer, j::Integer) = (x, y) -> K(x, y)[i, j]
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
    function gradient_x(k, x::AbstractVector, y::AbstractVector)
        r = DiffResults.GradientResult(y) # this could take pre-allocated temporary storage
        FD.gradient!(r, z->k(z, y), x)
        vcat(r.value, r.derivs[1]) # d+1 # ... to avoid this
    end
    value = @view K[:, 1] # value of helper, i.e. original value and gradient stacked
    derivs = @view K[:, 2:end] # jacobian of helper (higher order terms)
    result = DiffResults.DiffResult(value, derivs)
    ForwardDiff.jacobian!(result, z->gradient_x(g.k, x, z), y)
    return K
end

# 1d-input
function (k::GradientKernel)(x::Real, y::Real)
    evaluate!(zeros(typeof(x), (2, 2)), k, x, y)
end

function evaluate!(K::AbstractMatrix, g::GradientKernel, x::Real, y::Real)
    k = g.k
    dkx(x, y) = derivative((z)->k(z, y), x) # ∂k/∂x
    dkxy(x, y) = derivative((z)->dkx(x, z), y) # (∂k/∂x)/∂y = ∂k/(∂x∂y)
    dky(x, y) = derivative((z)->k(x, z), y) # k(x, y) = k(y, x) -> (∂k/∂x)(x,y)
    K[1, 1] = k(x, y)
    K[2, 1] = dkx(x, y)
    K[1, 2] = dky(x, y)
    K[2, 2] = dkxy(x, y)
    return K
end
