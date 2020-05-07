# using Zygote
# using Gauss
# import Gauss: nlml
# using Optimization

# TODO: pre-allocating dual numbers can greatly accelerate
# function extractDual!(dK::AbstractMatrix, dualK::AbstractMatrix)
#     @inbounds @simd for j in 1:length(dK)
#          dK[j] = dualK[j].partials[1]::eltype(dK) # extract derivative information
#     end
#     return
# end
using WoodburyIdentity
using LinearAlgebraExtensions: lowrank
using Optimization

include("../test/nlml.jl")

######################### Noise Variance Optimization ##########################
struct NoiseGradient{T, KT<:AbstractMatOrFac{T}, Y} <: Optimization.Direction{T}
    K::KT
    y::Y
end
function NoiseGradient(k::MercerKernel, x::AbstractVector, y::AbstractVector)
    K = gramian(k, x)
    K = lowrank(cholesky, K)
    NoiseGradient(K, y)
end
function Optimization.value(D::NoiseGradient, log_σ::AbstractVector)
    σ = exp(log_σ[1])
    Σ = σ^2*I(size(D.K, 1))
    W = Woodbury(Σ, D.K)
    v = safe_nlml(W, D.y)
    println("noise grad")
    println(v, log_σ)
    v
end
function Optimization.value_direction(D::NoiseGradient, log_σ::AbstractVector)
    σ = exp(log_σ[1])
    Σ = σ^2*I(size(D.K, 1))
    W = Woodbury(Σ, D.K)
    val, push = pushforward(nlml, W, D.y)
    dσ = similar(log_σ)
    dσ[1] = -2σ*push(I(size(D.K, 1)))
    dσ[1] *= σ # chain rule from exponential parameterization
    dσ[1] /= max(1, abs(dσ[1]))
    val, dσ
end
function optimize_noise(k::MercerKernel, x, y, σ::Real)
    θ = [log(σ)]
    Dir = NoiseGradient(k, x, y)
    Dir = Optimization.LBFGS(Dir, θ, 2)
    Dir = Optimization.DecreasingStep(Dir, θ)
    Optimization.fixedpoint!(Dir, θ)
    σ = exp(θ[1])
end#

####################### Kernel Hyper-Parameter Optimization ####################
# TODO: prior distribution for θ
struct KernelGradient{T, K<:MercerKernel{T}, X, Y,
                            S<:AbstractMatOrFac, P} <: Optimization.Direction{T}
    k::K
    x::X
    y::Y
    Σ::S
    θ::P # temporary storage for parameters
end
function KernelGradient(k, x, y, σ::Real = 1., θ = parameters(k))
    Σ = σ^2*I(length(x))
    KernelGradient(k, x, y, Σ, θ)
end

# TODO: more general parameterization?
parameterize!(θ::AbstractVector) = (@. θ = log(θ))
unparameterize!(θ::AbstractVector) = (@. θ = exp(θ))

function Optimization.value(D::KernelGradient, θ::AbstractVector)
    unparameterize!(θ)
    v = _value(D, θ)
    parameterize!(θ)
    println("kernel grad")
    println(v)
    v
end

function _value(D::KernelGradient, θ::AbstractVector)
    k = similar(D.k, θ)
    K = gramian(k, D.x)
    K = try lowrank(cholesky, K) catch e; return Inf end
    W = Woodbury(D.Σ, K)
    safe_nlml(W, D.y)
end

function Optimization.value_direction(D::KernelGradient, θ::AbstractVector)
    unparameterize!(θ)
    val, dir = _value_direction(D, θ)
    dir .*= θ # chain rule for exponential characterization
    parameterize!(θ)
    dir ./= norm(dir)
    val, dir
end

using Base.Threads
function _value_direction(D::KernelGradient, θ::AbstractVector)
    k = similar(D.k, θ)
    K = gramian(k, D.x)
    W = Woodbury(D.Σ, lowrank(cholesky, K))
    val, push = pushforward(nlml, W, D.y)
    dθ = similar(θ)
    @threads for i in 1:length(θ)
        dK = Kernel.derivative!_matrix(K, i)
        dθ[i] = push(dK)
    end
    dθ .*= -1 # want to take negative gradient step
    val, dθ
end

function optimize_kernel(k::MercerKernel, x::AbstractVector, y, σ::Real)
    θ = parameterize!(parameters(k))
    Dir = KernelGradient(k, x, y, σ, θ)
    # Dir = Optimization.LBFGS(Dir, θ, 7)
    Dir = Optimization.DecreasingStep(Dir, θ)
    Optimization.fixedpoint!(Dir, θ)
    unparameterize!(θ)
    similar(k, θ)
end

# optimize marginal likelihood w.r.t. hyper-parameters of MercerKernel
function optimize(k::MercerKernel, x::AbstractVector, y::AbstractVector, σ::Real;
    maxiter = 2)
    for i in 1:maxiter
        k = optimize_kernel(k, x, y, σ)
        σ = optimize_noise(k, x, y, σ)
    end
    k, σ
end

# returns matrix of ith partial derivatives
function derivative_matrix(G::Gramian, i::Integer)
    θ = parameters(G.k)
    n = length(θ)
    function f(α)
        p = zeros(typeof(α), n)
        p .= θ
        p[i] = α
        k = similar(G.k, p) # constructing kernel with dual parameter vector
        k.(G.x, permutedims(G.y))
    end
    FD.derivative(f, θ[i]) # TODO make lazy KernelMatrix
end

function derivative!_matrix(G::Gramian, i::Integer)
    θ = parameters(G.k)
    n = length(θ)
    function f(α)
        p = zeros(typeof(α), n)
        p .= θ
        p[i] = α
        k = similar(G.k, p) # constructing kernel with dual parameter vector
        k.(G.x, permutedims(G.y))
    end
    out = zeros(size(G))
    FD.derivative!(out, f, θ[i]) # TODO make lazy KernelMatrix
end

# elementwise gradient of kernel matrix for FiniteBasis
# TODO: deprecate
# function phasegradmat(P::Phase, x, θ::Vector)
#     length(θ) == nparams(P) || error("length(θ) ≠ nparams(P)")
#     val, Jac = phase_jacobian(P, x, θ)
#     # TODO: this could serve as basis for gradient definition of FiniteBasis gramian
#     begin # negligible cost
#         ∇ = [view(Jac, :, i) for i in 1:length(θ)]
#         ∇ = LowRank.(∇, (val',))
#         ∇ = ∇ .+ adjoint.(∇)     # d = LowRank(d, Qx) + LowRank(d, Qx)'
#     end
#     # println(rank.(∇)) == fill(2, length(θ)) # this could be a test
#     return ∇ # length(θ) rank two kernel gradient matrices
# end


############ negative log marginal likelihood gradient given factorization of Σ
# to allow for highler level autodifferentiation, use parametric types less here
# this should extend the Gauss nlml ...
# function Gauss.nlml(k::MercerKernel, x::AbstractVector, y::AbstractVector)
#     nlml(gramian(k, x), y)
# end
#
# function Gauss.nlml(k::MercerKernel, σ²::Real, x::AbstractVector, y::AbstractVector)
#     nlml(gramian(k, x) + σ²*I, y)
# end
#
# # TODO: do the intermediate step in reverse mode too? might not be that effective ...
# # val, Y = pullback(::typeof(Gauss.nlml), Σ, y)[1]
# # back(Y) = tr(Y'*dΣ) or tr(Y'.*dΣ) if dΣ is array of matrices ...
#
# # TODO: preallocate memory, where possible
# # more like pullback(::Tuple{typeof(nlml), k, <:AbstractVector, <:AbstractVector}, θ)
# function pullback(::typeof(Gauss.nlml), k, θ::Tuple, x::AbstractVector, y::AbstractVector)
#     Σ = gramian(k(θ...), x)
#     Σ = factorize(Σ) # maybe this is not necessary
#     val, forward = Gauss.pushforward(nlml, Σ, y)
#     grad = gradient(k, θ, x)
#     return val, ∇ -> ∇ .* tuple((forward(dΣ) for dΣ in grad)...)
# end
#
# # approximate second derivative calculation
# function approx_laplacian(::typeof(nlml), k, θ::VT, x::V, y::U,
#                             Σ::S;
#                             δ::T = T(1e-8)) where {T, VT<:AbstractVector,
#                             V<:AbstractArray{<:Union{T, AbstractArray{T}}},
#                             U<:AbstractArray{T}, S<:Factorization{T}}
#     n = length(x)
#     Dual = FD.Dual{FD.Tag{typeof(k), T}}
#     θ_d = Dual.(θ, 0.)
#     ∇ = zeros(size(θ))
#     Δ = zeros(size(θ))
#     dualK = zeros(FD.Dual{FD.Tag{typeof(k), T}, T, 1}, (n, n))
#     dK = zeros(T, (n,n))
#     # could also try batched gradient calculation at the expense of memory consumption
#     for i in 1:length(θ_d)
#         θ_d .= Dual.(θ, 0.) # reset dual array
#         θ_d[i] += Dual(0., 1.) # add dual component to ith parameter
#         dualK .= k(θ_d).(x, permutedims(x)) # evaluates kernel matrix and partial derivative
#         # dualK .= gramian(k(θ_d), x)
#         extractDual!(dK, dualK)
#          # most expensive operation:
#         ∇[i], Δ[i] = Gauss.approx_laplacian(nlml, Σ, dK, y)
#     end
#     return ∇, Δ
# end
#
# function gradient(::typeof(nlml), k, θ, x::AbstractVector, y::AbstractVector,
#                             σ_prior, d::Real = 1e-8)
#     ∇, Δ = approx_laplacian(nlml, k, θ, x, y)
#     # ∇ .+= ForwardDiff.gradient(prior, θ)
#     # Δ .+= ForwardDiff.gradient(prior, θ) # need to get laplacian here
#     return @. ∇ / (abs(Δ) + d)
# end


################### Memory efficient gradient calculation
# more memory efficient call
# function gradient!(::typeof(nlml),
#                 ∇_Θ::VT, ∇_k::M, dualK::MD,
#                 k, θ::VT, Σ::S,
#                 x::V, y::U) where {T, M <: AbstractMatrix,
#                                     MD <: AbstractMatrix,
#                                     VT<:AbstractArray,
#                                     S<:Factorization{T},
#                                     V<:AbstractArray{<:Union{T, AbstractArray{T}}},
#                                     U<:AbstractArray{T}}
#
#     # k should be a parameterized family of kernels k(θ)::MercerKernel
#     # e.g.: k(θ) = θ[1] * Kernel.EQ()
#     n = length(x)
#
#     # pre-allocate the next four lines
#     Dual = FD.Dual{FD.Tag{typeof(k), T}}
#     θ_d = Dual.(Θ, 0.)
#     # ∇ = zeros(size(Θ))
#     # dualK = zeros(FD.Dual{FD.Tag{typeof(kf), T}, T, 1}, (n, n))
#
#     for i = 1:length(θ_d)
#         θ_d .= Dual.(Θ, 0.) # reset dual array
#         θ_d[i] += Dual(0., 1.) # add dual component to ith parameter
#         dualK .= kf(θ_d).(x, permutedims(x)) # evaluates kernel matrix and partial derivative
#         for i = 1:length(K)
#                ∇_k[i] = dualK[i].partials[1] # extract derivative information
#         end
#         ∇_Θ[i] = gradient(nlml)(Σ, ∇_k, y) # calculate derivative w.r.t. Θ[i]
#     end
#     return ∇_Θ
# end

# function gradient!(::typeof(nlml), ∇, θ, θ_d, Σ, dK, dualK, x, y)
#     # could also try batched gradient calculation at the expense of memory consumption
#     for i = 1:length(θ_d)
#         θ_d .= Dual.(θ, 0.) # reset dual array
#         θ_d[i] += Dual(0., 1.) # add dual component to ith parameter
#         dualK .= k(θ_d).(x, permutedims(x)) # evaluates kernel matrix and partial derivative
#         extractDual!(dK, dualK)
#          # most expensive operation:
#         # ∇[i] = Gauss.gradient(nlml, Σ, dK, y) # calculate derivative w.r.t. Θ[i]
#         ∇[i] = Gauss.scaled_gradient(nlml, Σ, dK, y) # calculate derivative w.r.t. Θ[i]
#     end
#     return
# end
