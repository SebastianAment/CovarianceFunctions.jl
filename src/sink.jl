using LinearAlgebra
using Distributions
using StatsFuns
using Kernel

# A collection of random kitchen sink methods from
# https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf
# https://people.eecs.berkeley.edu/~brecht/papers/08.Rah.Rec.Allerton.pdf
# https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

function random_features(ϕ, x, wdist, bdist, D; islinear=Val(true))
    w = rand(wdist, D, size(x, 1))
    b = rand(bdist, x, size(x, 1))
    return random_features(ϕ, x, w, b, D; islinear), w, b
end

function random_features(ϕ, x, w::Array, b::Array, D; islinear)
    ϕx = similar(x, D, size(x, 2))
    return random_features!(ϕx, ϕ, x, w, b; islinear)
end

function random_features!(ϕx, ϕ, x, w, b; islinear::Val{false})
    d = size(ϕx, 2)
    colx = eachcol(x)
    for i in 1:d
        @views ϕx[i, :] .= (y -> ϕ(y, w[:, i]), b).(colx)
    end
    return ϕx
end

function random_features!(ϕx, ϕ, x, w, b; islinear::Val{true})
    mul!(ϕx, w, x)
    ϕx .= ϕ.(ϕx .+ b)
    return ϕx
end

function fourier_dist(k)
end

function fourier_features(k, x, D)
    bdist = Uniform(0, twoπ)
    wdist = fourier_dist(k)
    ϕ = x -> sqrt(2 / D) * cos(x) 
    return random_features(ϕ, x, wdist, bdist, D)
end

include("./sink_lr.jl")
include("./sink_svm.jl")
