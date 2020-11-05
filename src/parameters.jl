################################################################################
# this section automates the construction of parameter-dependent composite kernels
# parameters function returns either scalar or recursive vcat of parameter vectors
# useful for optimization algorithms which require the parameters in vector form
parameters(::Any) = []
parameters(::AbstractKernel{T}) where {T} = zeros(T, 0)
nparameters(::Any) = 0

# checks if θ has the correct number of parameters to initialize a kernel of typeof(k)
function checklength(k::AbstractKernel, θ::AbstractVector)
    nt = length(θ)
    np = nparameters(k)
    if nt ≠ np
        throw(DimensionMismatch("length(θ) = $nt ≠ $np = nparameters(k)"))
    end
    return nt
end

# thanks to ffevotte in https://discourse.julialang.org/t/how-to-call-constructor-of-parametric-family-of-types-efficiently/38503/5
stripped_type(x) = stripped_type(typeof(x))
stripped_type(typ::DataType) = Base.typename(typ).wrapper
# fallback for zero-parameter kernels
function Base.similar(k::AbstractKernel, θ::AbstractVector)
    n = checklength(k, θ)
    # kernel = eval(Meta.parse(string(typeof(k).name)))
    kernel = stripped_type(k)
    if n == 0
        kernel()
    elseif n == 1
        kernel(θ[1])
    else
        kernel(θ)
    end
end

Base.similar(k::AbstractKernel, θ::Number) = similar(k, [θ])

# constructs a similar kernel to k with hyperparameters θ
function _similar_helper(k, θ)
    checklength(k, θ)
    args = Vector{AbstractKernel}(undef, length(k.args))
    for (i, k) in enumerate(k.args)
        n = nparameters(k)
        args[i] = similar(k, @view(θ[1:n]))
        θ = @view(θ[n+1:end])
    end
    return args
end
