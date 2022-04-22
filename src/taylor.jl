####################### core taylor expansion based algorithm ##################
# still based on BarnesHutFactorization, should probably introduce new TaylorFactorization type
# use_com = true Taylor expands around the center of mass,
# improves on accuracy of BarnesHut if weight vectors have negative elements
# Generally, taylor! is ~2x faster and slightly more accurate than splitting_barneshut!
# when a signficiant degree of compression is taking place (i.e larger θ)
function taylor!(b::AbstractVector, F::BarnesHutFactorization{<:Number}, w::AbstractVector,
                    α::Number = 1, β::Number = 0, θ::Real = F.θ;
                    split::Bool = true, use_com::Bool = true)
    size(F, 2) == length(w) || throw(DimensionMismatch("length of w does not match second dimension of F: $(length(w)) ≠ $(size(F, 2))"))
    eltype(b) == promote_type(eltype(F), eltype(w)) ||
            throw(TypeError("eltype of target vector b not equal to eltype of matrix-vector product: $(eltype(b)) and $(promote_type(eltype(F), eltype(w)))"))
    if β == 0
        @. b = 0 # this avoids trouble if b is initialized with NaN's, e.g. thorugh "similar"
    else
        @. b *= β
    end
    f0 = _derivative_helper(F.k)
    f1(r²::Real) = ForwardDiff.derivative(f0, r²)
    f_orders = (f0, f1)
    # or functional form: f_orders(r) = value_derivative(f0, r)

    sums_w = node_sums(w, F.Tree) # IDEA: could pre-allocate
    sums_w_r = weighted_node_sums(F.y, w, F.Tree)
    centers = use_com ? compute_centers_of_mass(F, w) : get_hyper_centers(F)
    @. sums_w_r -= sums_w * centers # need to center the moments
    @threads for i in eachindex(F.x) # exactly 4 * length(y) allocations?
        Fw_i = taylor_recursion(1, F.k, f_orders, F.x[i], F.y, w, sums_w, sums_w_r, θ, F.Tree, centers)::eltype(b) # x[i] creates an allocation
        b[i] += α * Fw_i
    end
    if !isnothing(F.D) # if there is a diagonal correction, need to add it
        mul!(b, F.D, w, α, 1)
    end

    return b
end

# k_orders returns tuple of kernel derivatives up to pth order

# uses taylor expansion as approximation (identical with bh_recursion if center is center of mass)
function taylor_recursion(index, k, f_orders, xi, y::AbstractVector, w::AbstractVector,
                      sums_w::AbstractVector, sums_w_r::AbstractVector, θ::Real, T::BallTree, centers)
    h = T.hyper_spheres[index]
    if isleaf(T.tree_data.n_internal_nodes, index) # do direct computation
        elty = promote_type(eltype(xi), eltype(eltype(y)), eltype(w))
        val = zero(elty)
        @inbounds @simd for i in get_leaf_range(T.tree_data, index) # loop vectorization?
            j = T.indices[i]
            val += k(xi, y[j]) * w[j]
        end
        return val

    elseif h.r < θ * euclidean(xi, centers[index]) # compress
        ri = difference(xi, centers[index])
        sum_abs2_ri = sum(abs2, ri)
        f0, f1 = f_orders # contains first and second order derivative evaluation
        value = f0(sum_abs2_ri) * sums_w[index]
        first_order = -2*f1(sum_abs2_ri) * dot(ri, sums_w_r[index])
        # second_order ...
        return value + first_order

    else # recurse NOTE: parallelizing here is not as efficient as parallelizing over target points
        l = taylor_recursion(getleft(index), k, f_orders, xi, y, w, sums_w, sums_w_r, θ, T, centers)
        r = taylor_recursion(getright(index), k, f_orders, xi, y, w, sums_w, sums_w_r, θ, T, centers)
        return l + r
    end
end

# this represents the Array with values r[i_1]*r[i_2]*...*r[i_p] lazily
# it will be useful for fast approximate matrix multiplication based on
# higher order taylor expansions
struct PowersArray{T, X <: AbstractVector{T}, P} <: AbstractArray{T, P}
    x::X
end

function PowersArray(x, p::Int)
    p ≥ 1 || throw(DomainError("Can't construct PowersArray with p < 1"))
    PowersArray{eltype(x), typeof(x), p}(x)
end
power(A::PowersArray{<:Any, <:Any, P}) where P = P

Base.length(A::PowersArray) = length(A.x)^power(A)
Base.size(A::PowersArray) = tuple((size(A, 1) for _ in 1:power(A))...)
function Base.size(A::PowersArray, i::Int)
    1 ≤ i ≤ power(A) ? length(A.x) : 1
end

function Base.getindex(A::PowersArray{<:Any, <:Any, P}, indices::Vararg{Int, P}) where P
    r = one(eltype(A.x))
    for i in indices
        r *= A.x[i]
    end
    return r
end
