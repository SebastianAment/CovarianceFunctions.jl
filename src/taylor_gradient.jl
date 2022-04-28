############################### matrix valued version ##########################
# function BarnesHutFactorization(k::GradientKernel, x, y = x, D = nothing;
#                     θ::Real = 1/4, leafsize::Int = BARNES_HUT_DEFAULT_LEAFSIZE)
#     xs = vector_of_static_vectors(x)
#     ys = x === y ? xs : vector_of_static_vectors(y)
#     Tree = BallTree(ys, leafsize = leafsize)
#     m = length(y)
#     XT, YT, KT, TT, DT, RT = typeof.((xs, ys, k, Tree, D, θ))
#     # w = zeros(length(m))
#     # i = zeros(Bool, m)
#     # WT, BT = typeof(w), typeof(i)
#     T = gramian_eltype(k, xs, ys)
#     F = BarnesHutFactorization{T, KT, XT, YT, TT, DT, RT}(k, xs, ys, Tree, D, θ) #, w, i)
# end

function LinearAlgebra.mul!(y::AbstractVector{<:Real}, F::BarnesHutFactorization{<:Any, <:GradientKernel},
                            x::AbstractVector{<:Real}, α::Real = 1, β::Real = 0)
    d = length(F.x[1])
    X, Y = reshape(x, d, :), reshape(y, d, :) # converting vector of reals to vector of vectors
    xx, yy = [c for c in eachcol(X)], [c for c in eachcol(Y)]
    mul!(yy, F, xx, α, β)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{<:AbstractVector}, F::BarnesHutFactorization{<:Any, <:GradientKernel},
                            x::AbstractVector{<:AbstractVector}, α::Real = 1, β::Real = 0)
    taylor!(y, F, x, α, β)
end

# function Base.:*(F::BarnesHutFactorization{<:Number}, x::AbstractVector{<:Number})
#     T = promote_type(eltype(F), eltype(x))
#     y = zeros(T, size(F, 1))
#     mul!(y, F, x)
# end

function taylor!(b::AbstractVector{<:AbstractVector}, F::BarnesHutFactorization{<:Any, <:GradientKernel},
                 w::AbstractVector{<:AbstractVector}, α::Number = 1, β::Number = 0, θ::Real = F.θ)
    size(F, 2) == length(w) || throw(DimensionMismatch("length of w does not match second dimension of F: $(length(w)) ≠ $(size(F, 2))"))
    # eltype(b) == promote_type(eltype(F), eltype(w)) ||
    #         throw(TypeError("eltype of target vector b not equal to eltype of matrix-vector product: $(eltype(b)) and $(promote_type(eltype(F), eltype(w)))"))
    f_orders(r²) = derivatives(F.k.k, r², 3)
    sums_w = node_sums(w, F.Tree) # IDEA: could pre-allocate, worth it? is several orders of magnitude less expensive than multiply
    sums_w_r = weighted_node_sums(adjoint.(w), adjoint.(F.y), F.Tree) # need sum of outer products of F.y and w
    centers = get_hyper_centers(F)
    @. sums_w_r -= sums_w * adjoint(centers) # need to center the moments
    Gijs = [F[1, 1] for _ in 1:Base.Threads.nthreads()]
    for i in eachindex(F.x) # exactly 4 * length(y) allocations?
        if β == 0
            @. b[i] = 0 # this avoids trouble if b is initialized with NaN's, e.g. thorugh "similar"
        else
            @. b[i] *= β
        end
        Gij = Gijs[Base.Threads.threadid()]
        taylor_recursion!(b[i], Gij, 1, F.k, f_orders, F.x[i], F.y,
                          w, sums_w, sums_w_r, θ, F.Tree, centers, α) # x[i] creates an allocation
    end
    if !isnothing(F.D) # if there is a diagonal correction, need to add it
        mul!(b, F.D, w, α, 1)
    end
    return b
end

# barnes hut recursion for matrix-valued kernels, could merge with scalar version
# bi is target vector corresponding to input point xi
# Gij is temporary storage for evaluation of k(xi, y[j]), important if it is matrix valued
# to avoid allocations
function taylor_recursion!(bi::AbstractVector, Gij,
                      index, k::GradientKernel, f_orders,
                      xi, y::AbstractVector,
                      w::AbstractVector{<:AbstractVector},
                      sums_w::AbstractVector{<:AbstractVector},
                      sums_w_r::AbstractVector{<:AbstractMatrix},
                      θ::Real, T::BallTree, centers, α::Number)
    h = T.hyper_spheres[index]
    if isleaf(T.tree_data.n_internal_nodes, index) # do direct computation
        for i in get_leaf_range(T.tree_data, index)
            j = T.indices[i]
            # @time Gij = evaluate_block!(Gij, k, xi, y[j]) # k(xi, y[j])
            Gij = IsotropicGradientKernelElement{eltype(bi)}(k.k, xi, y[j])
            wj = w[j]
            mul!(bi, Gij, wj, α, 1)
        end
        return bi

    elseif h.r < θ * euclidean(xi, centers[index]) # compress
            S_index = sums_w_r[index]
            ri = difference(xi, centers[index])
            sum_abs2_ri = sum(abs2, ri)
            # NOTE: this line is the only one that still allocates ~688 bytes
            f0, f1, f2, f3 = f_orders(sum_abs2_ri) # contains first and second order derivative evaluation

            # Gij = evaluate_block!(Gij, k, xi, centers[index]) # k(xi, centers_of_mass[index])
            Gij = IsotropicGradientKernelElement{eltype(bi)}(k.k, xi, centers[index])

            # zeroth order
            mul!(bi, Gij, sums_w[index], α, 1) # bi .+= α * k(xi, centers_of_mass[index]) * sums_w[index]
            # first order
            # this block has zero allocations
            mul!(bi, 2*f3*dot(ri, S_index, ri) + f2 * tr(S_index), ri, 4α, 1)
            mul!(bi, S_index, ri, 4α*f2, 1)
            mul!(bi, S_index', ri, 4α*f2, 1)
        return bi
    else # recurse NOTE: parallelizing here is not as efficient as parallelizing over target points
        taylor_recursion!(bi, Gij, getleft(index), k, f_orders, xi, y, w, sums_w, sums_w_r, θ, T, centers, α)
        taylor_recursion!(bi, Gij, getright(index), k, f_orders, xi, y, w, sums_w, sums_w_r, θ, T, centers, α)
    end
end

# function node_mapreduce(x::AbstractVector, w::AbstractVector, T::BallTree, index::Int = 1)
#     length(x) == 0 && return zero(eltype(x))
#     sums = fill(w[1]*x[1]', length(T.hyper_spheres))
#     node_outer_products!(sums, x, w, T)
# end
#
# # NOTE: x should either be vector of numbers or vector of static arrays
# function node_mapreduce!(sums::AbstractVector{<:AbstractMatrix}, x::AbstractVector,
#                             w::AbstractVector{<:Number}, T::BallTree, index::Int = 1)
#     if isleaf(T.tree_data.n_internal_nodes, index)
#         i = get_leaf_range(T.tree_data, index)
#         wi, xi = @views w[T.indices[i]], x[T.indices[i]]
#         sums[index] = wi * xi'
#         # adjoint.(w)' * adjoint.(x)
#     else
#         task = @spawn weighted_node_sums!(sums, x, w, T, getleft(index))
#         weighted_node_sums!(sums, x, w, T, getright(index))
#         wait(task)
#         sums[index] = sums[getleft(index)] + sums[getright(index)]
#     end
#     return sums
# end
