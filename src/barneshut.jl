using NearestNeighbors: TreeData, get_leaf_range, isleaf, getleft, getright

const BARNES_HUT_DEFAULT_LEAFSIZE = 16

# could generalize to something like HierarchicalFactorization
# have field that determines algorithm to execute Taylor(p), BarnesHut(), FKT(p)
struct BarnesHutFactorization{T, KT, XT<:AbstractVector, YT<:AbstractVector, TT, DT, RT} <: Factorization{T}
    k::KT # kernel function, for taylor branch, needs to be evaluable as k(r²)
    x::XT
    y::YT
    Tree::TT
    D::DT # diagonal correction (i.e. noise variance for GPs)
    θ::RT # distance threshold for compression
    # expansion_type::ET # IDEA to generalize tree-based algorithms: problem: would be nice to pre-allocate moments
    # w::WT # temporary storage for weight vectors, useful to split positive and negative values
    # i::BT
    # centers_of_mass::CT # pre-allocated, has to be able to work with type of w
end

# NOTE on ball tree indices
# if reorder = true, and T = BallTree(X), we have
# X[:, T.indices[i]] = T.data[i]
# X[:, i] == T.data[invpermute!(collect(1:length(T.data)), T.indices)][i]
function BarnesHutFactorization(k, x, y = x, D = nothing; θ::Real = 1/4, leafsize::Int = BARNES_HUT_DEFAULT_LEAFSIZE)
    xs = vector_of_static_vectors(x)
    ys = x === y ? xs : vector_of_static_vectors(y)
    Tree = BallTree(ys, leafsize = leafsize)
    m = length(y)
    XT, YT, KT, TT, DT, RT = typeof.((xs, ys, k, Tree, D, θ))
    # w = zeros(length(m))
    # i = zeros(Bool, m)
    # WT, BT = typeof(w), typeof(i)
    T = gramian_eltype(k, xs[1], ys[1])
    BarnesHutFactorization{T, KT, XT, YT, TT, DT, RT}(k, xs, ys, Tree, D, θ) #, w, i)
end
function BarnesHutFactorization(G::Gramian, θ::Real = 1/2; leafsize::Int = BARNES_HUT_DEFAULT_LEAFSIZE)
    BarnesHutFactorization(G.k, G.x, G.y, θ, leafsize = leafsize)
end
Base.size(F::BarnesHutFactorization) = (length(F.x), length(F.y))
Base.size(F::BarnesHutFactorization, i::Int) = i ≤ 2 ? size(F)[i] : 1
Base.getindex(F::BarnesHutFactorization, i::Int, j::Int) = F.k(F.x[i], F.y[j])
Base.eltype(F::BarnesHutFactorization{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, F::BarnesHutFactorization, x::AbstractVector, α::Real = 1, β::Real = 0)
    # barneshut!(y, F, x, α, β)
    if all(≥(0), x) # if all weights are positive, proceed with regular BarnesHut
        barneshut!(y, F, x, α, β, split = false)
    else
        taylor!(y, F, x, α, β)
    end
end
function Base.:*(F::BarnesHutFactorization{<:Number}, x::AbstractVector{<:Number})
    T = promote_type(eltype(F), eltype(x))
    y = zeros(T, size(F, 1))
    mul!(y, F, x)
end

# # use cg! if it's positive definite
# keywords include maxiter = size(F, 2)
# verbose: convergence info
# log: keep track of residual norm
# Pl, Pr: left and right preconditioners (not implemented?)
function LinearAlgebra.ldiv!(x::AbstractVector, F::BarnesHutFactorization, b::AbstractVector; kwargs...)
    # cg!(x, F, b; kwargs...)
    minres!(x, F, b; kwargs...)
end
function Base.:\(F::BarnesHutFactorization, b::AbstractVector; kwargs...)
    T = promote_type(eltype(F), eltype(b))
    x = zeros(T, size(F, 2))
    ldiv!(x, F, b; initially_zero = true, kwargs...)
end

############################ Core Barnes Hut algorithm ##########################
# uses the Barnes-Hut algorithm for an approximate n*log(n) multiply with the kernel matrix G
function barneshut!(b::AbstractVector, F::BarnesHutFactorization{<:Number}, w::AbstractVector,
                    α::Number = 1, β::Number = 0, θ::Real = F.θ; split::Bool = true)
    size(F, 2) == length(w) || throw(DimensionMismatch("length of w does not match second dimension of F: $(length(w)) ≠ $(size(F, 2))"))
    eltype(b) == promote_type(eltype(F), eltype(w)) ||
            throw(TypeError("eltype of target vector b not equal to eltype of matrix-vector product: $(eltype(b)) and $(promote_type(eltype(F), eltype(w)))"))

    if β == 0
        @. b = 0 # this avoids trouble if b is initialized with NaN's, e.g. thorugh "similar"
    else
        @. b *= β
    end

    if split && any(<(0), w) # if split is on, we multiply with positive and negative component of w separately, tends to increase accuracy because of center of mass calculation
        splitting_barneshut!(b, F, w, α, β, θ)
    else
        sums_w = node_sums(w, F.Tree) # IDEA: could pre-allocate
        centers_of_mass = compute_centers_of_mass(F, w)
        @threads for i in eachindex(F.x) # exactly 4 * length(y) allocations?
            Fw_i = bh_recursion(1, F.k, F.x[i], F.y, w, sums_w, θ, F.Tree, centers_of_mass)::eltype(b) # x[i] creates an allocation
            b[i] += α * Fw_i
        end
        if !isnothing(F.D) # if there is a diagonal correction, need to add it
            mul!(b, F.D, w, α, 1)
        end
    end
    return b
end

# splits weight vector a into positive and negative components, tends to have
# much better accuracy for negative a, for only ~2x time penalty
function splitting_barneshut!(b::AbstractVector, F::BarnesHutFactorization, a::AbstractVector,
                    α::Number = 1, β::Number = 0, θ::Real = F.θ)
    c = copy(a) # IDEA: pre-allocate
    i = a .< 0 # first, blend out negative indices (multiply with positive part)
    @. c[i] = 0
    barneshut!(b, F, c, α, β, θ, split = false)
    @. c = -a # multiply with -a[i] to make entries positive, and use negative α to make result correct
    @. i = !i # blend out non-negative indices
    @. c[i] = 0
    barneshut!(b, F, c, -α, 1, θ, split = false) # use β = 1 because β is already taken care of above
    return b
end

# supposing y is a single value now
# k sould be isotropic kernel with euclidean norm
# x is single target data point
# y are all source data points
# sums_a[index] contains sum(a[get_node_indices(T.tree_data, index)])
# θ < 1 is the distance threshold beyond which compression is used
# e.g. when θ = 1/10, then we compress when the current target node is 10 times further
# from the center of the hypersphere than its radius
# when θ = 1, everything is compressed, when θ = 0, nothing is
function bh_recursion(index, k, xi, y::AbstractVector, w::AbstractVector,
                      sums_w::AbstractVector, θ::Real, T::BallTree, centers_of_mass)
    h = T.hyper_spheres[index]
    if isleaf(T.tree_data.n_internal_nodes, index) # do direct computation
        elty = promote_type(eltype(xi), eltype(eltype(y)), eltype(w))
        val = zero(elty)
        @inbounds @simd for i in get_leaf_range(T.tree_data, index) # loop vectorization?
            j = T.indices[i]
            val += k(xi, y[j]) * w[j]
        end
        return val

    elseif h.r < θ * euclidean(xi, centers_of_mass[index]) # compress
        return k(xi, centers_of_mass[index]) * sums_w[index]

    else # recurse NOTE: parallelizing here is not as efficient as parallelizing over target points
        l = bh_recursion(getleft(index), k, xi, y, w, sums_w, θ, T, centers_of_mass)
        r = bh_recursion(getright(index), k, xi, y, w, sums_w, θ, T, centers_of_mass)
        return l + r
    end
end

############################# centers of mass ##################################
# this is a weighted sum, could be generalized to incorporate node_sums
function compute_centers_of_mass(w::AbstractVector, x::AbstractVector, T::BallTree)
    D = eltype(x) <: StaticVector ? length(eltype(x)) : length(x[1]) # if x is static vector
    com = [zero(MVector{D, Float64}) for _ in 1:length(T.hyper_spheres)]
    compute_centers_of_mass!(com, w, x, T)
end

function compute_centers_of_mass(F::BarnesHutFactorization, w::AbstractVector)
    compute_centers_of_mass(w, F.y, F.Tree)
end

function compute_centers_of_mass!(com::AbstractVector, w::AbstractVector, x::AbstractVector, T::BallTree)
    abs_w = abs.(w)
    weighted_node_sums!(com, abs_w, x, T)
    sum_w = node_sums(abs_w, T)
    ε = eps(eltype(w)) # ensuring division by zero it not a problem
    @. com ./= sum_w + ε
end

node_sums(x::AbstractVector, T::BallTree) = weighted_node_sums(Ones(length(x)), x, T)
function node_sums!(sums, x::AbstractVector, T::BallTree)
    weighted_node_sums!(sums, Ones(length(x)), x, T)
end

function weighted_node_sums(w::AbstractVector, x::AbstractVector, T::BallTree, index::Int = 1)
    length(x) == 0 && return zero(eltype(x))
    sums = fill(zero(w[1]'x[1]), length(T.hyper_spheres))
    weighted_node_sums!(sums, w, x, T)
end

# NOTE: x should either be vector of numbers or vector of static arrays
function weighted_node_sums!(sums::AbstractVector, w::AbstractVector,
                            x::AbstractVector, T::BallTree, index::Int = 1)
    if isleaf(T.tree_data.n_internal_nodes, index)
        i = get_leaf_range(T.tree_data, index)
        wi, xi = @views w[T.indices[i]], x[T.indices[i]]
        sums[index] = wi'xi
    else
        task = @spawn weighted_node_sums!(sums, w, x, T, getleft(index))
        weighted_node_sums!(sums, w, x, T, getright(index))
        wait(task)
        sums[index] = sums[getleft(index)] + sums[getright(index)]
    end
    return sums
end

# get centers used for compression in taylor factorization
function get_hyper_centers(F::BarnesHutFactorization)
    [hyp.center for hyp in F.Tree.hyper_spheres] # for uniform weights, doesn't loose accuracy in doing this
end
