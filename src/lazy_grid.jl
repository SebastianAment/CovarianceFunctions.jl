##################### Lazy multi-dimensional grid ##############################
# useful to automatically detect Kronecker structure in Kernel matrices
struct LazyGrid{T, V<:Tuple{Vararg{AbstractVecOrMat{T}}}} <: AbstractVector{Vector{T}}
    args::V
end
Base.length(G::LazyGrid) = prod(length, G.args)
Base.eltype(G::LazyGrid{T}) where {T} = Vector{T}
Base.size(G::LazyGrid) = (length(G),)
function Base.ndims(G::LazyGrid)
    sum(x -> x isa AbstractVector ? 1 : size(x, 1), G.args)
end
# default column-major indexing
function Base.getindex(G::LazyGrid{T}, i::Integer) where {T}
    @boundscheck checkbounds(G, i)
    val = zeros(T, ndims(G))
    n = length(G)
    d = ndims(G)
    @inbounds for a in reverse(G.args)
        n ÷= length(a)
        ind = cld(i, n) # can this be replaced with fld1, mod1?
        if a isa AbstractVector
            val[d] = a[ind]
            d -= 1
        else
            val[d-size(a, 1)+1:d] = a[:, ind]
            d -= size(a, 1)
        end
        i = mod1(i, n) # or some kind of shifted remainder?
    end
    return val
end
# row major indexing
function Base.getindex(G::LazyGrid{T}, i::Integer, rowmajor::Val{true}) where {T}
    @boundscheck checkbounds(G, i)
    val = zeros(T, ndims(G))
    n = length(G)
    d = 1
    @inbounds for (j, a) in enumerate(G.args)
        n ÷= length(a)
        ind = cld(i, n) # can this be replaced with fld1, mod1?
        if a isa AbstractVector
            val[d] = a[ind]
            d += 1
        else
            val[d:d+size(a, 1)-1] = a[:, ind]
            d += size(a, 1)
        end
        i = mod1(i, n) # or some kind of shifted remainder?
    end
    return val
end

LazyGrid{T}(args) where {T} = LazyGrid{T, typeof(args)}(args)A
LazyGrid(args) = LazyGrid{Float64}(args)
grid(args::Tuple) = LazyGrid(args)
grid(args::AbstractArray...) = grid(args)
