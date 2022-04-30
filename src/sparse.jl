# sparsification algorithm for exponentially decaying isotropic kernel matrices in high dimensions in particular
using NearestNeighbors
const SPARSE_DEFAULT_LEAFSIZE = 16

function SparseArrays.sparse(G::Gramian, δ::Real = 1e-6; leafsize::Int = SPARSE_DEFAULT_LEAFSIZE)
    n, m = size(G)
    if m < n # sparse(G) is faster if G is fat, rather than tall
        return sparse(sparse(G')')
    else
        K = spzeros(eltype(G), n, m)
        r = decay_radius(G.k, δ)
        neighbors = in_range_neighbors(G.x, G.y, r, leafsize = leafsize) # Bottleneck IDEA: do this on demand
        # Tree = BallTree(X, leafsize = leafsize)
        for j in 1:m
            # neighbors = inrange(Tree, G.y[j])
            for i in neighbors[j]
                K[i, j] = G.k(G.x[i], G.y[j]) # NOTE: insertion into sparse matrix is not thread-safe, would need to be atomic
            end
        end
        return K # IDEA: G.x === G.y ? Symmetric(K) : K
    end
end

# decay_radius calculates the isotropic radius beyond which a kernel is below δ
decay_radius(::ExponentiatedQuadratic, δ::Real) = sqrt(-2log(δ))
decay_radius(::Exponential, δ::Real) = -log(δ)
decay_radius(k::GammaExponential, δ::Real) = (-2log(δ))^(1/k.γ)
function decay_radius(k::Matern, δ::Real)
    if k.ν >= 1/2
        -log(δ)
    else
        throw(DomainError("decay_radius not defined for Matern kernel with ν < 1/2")) # this is a conservative estimate
    end
end
decay_radius(k::MaternP, δ::Real) = -log(δ) # this is a conservative estimate, could get tighter one with Taylor error bound of exponential in Matern formula
decay_radius(k) = throw(TypeError("decay_radius only defined for IsotropicKernel types, input is of type $typeof(k)"))

decay_radius(k::Lengthscale, δ::Real) = decay_radius(k.k) / k.l
# decay_radius(k::Normed, δ::Real) = -1

########
function in_range_neighbors(x, r::Real, metric = Euclidean(); leafsize::Int = DEFAULT_LEAFSIZE)
    in_range_neighbors(x, x, r, metric; leafsize = leafsize)
end

# calculates all neighbors in X to points in Y that are within a radius r using metric for distance evaluation
# reorder copies the data and puts nearby points close in memory, think about using Distance.jl for the rest of the code too
function in_range_neighbors(x, y, r::Real, metric = Euclidean(); leafsize::Int = DEFAULT_LEAFSIZE)
    # X needs to be either matrix or vector of static vectors
    x = vector_of_static_vectors(x)
    y = x === y ? x : vector_of_static_vectors(y)
    tree = BallTree(x, metric; leafsize = leafsize, reorder = true)
    inrange(tree, y, r) # plus one because we are putting in the original points as well
end
