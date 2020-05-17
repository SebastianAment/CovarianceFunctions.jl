
struct Woodbury
    U::Matrix{Float64}
    λ::Float64
    core::Cholesky{Float64, Matrix{Float64}}
    function Woodbury(U, λ)
        M = Matrix(I, size(U, 2), size(U, 2))
        mul!(M, U', U, 1 / λ, 1)
        return new(U, λ, cholesky!(M))
    end
end

function ldiv!(W::Woodbury, b)
    b .= b ./ W.λ .+ W.U * (W.core \ (W.U * b)) ./ W.λ ^ 2
    return b
end

function kitchen_woodbury(ϕ, x, wdist, bdist, D, λ)
    ϕx, w, b = random_features(ϕ, x, wdist, bdist, D)
    return Woodbury(ϕx, λ), ϕx, w, b
end

kitchen_lr(W, y) = W \ (W.U * y)