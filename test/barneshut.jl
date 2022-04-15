module TestBarnesHut
using LinearAlgebra
using CovarianceFunctions
using CovarianceFunctions: BarnesHutFactorization, barneshut!, vector_of_static_vectors,
        node_sums, euclidean, GradientKernel
using NearestNeighbors
using NearestNeighbors: isleaf, getleft, getright, get_leaf_range
using Test

function barneshut_no_far_field!(b::AbstractVector, F::BarnesHutFactorization, w::AbstractVector,
                    α::Number = 1, β::Number = 0, θ::Real = F.θ)
    D = length(eltype(F.x))
    sums_a = node_sums(w, F.Tree)
    # centers_of_mass = compute_centers_of_mass(F.y, w, F.Tree)
    centers_of_mass = [hyp.center for hyp in F.Tree.hyper_spheres]
    for i in eachindex(F.x)
        Fa_i = bh_recursion_no_far_field(1, F.k, F.x[i], F.y, w, sums_a, θ, F.Tree, centers_of_mass)
        b[i] = α * Fa_i + β * b[i]
    end
    return b
end

function bh_recursion_no_far_field(index, k, x, y::AbstractVector,
            w::AbstractVector, sums_w::AbstractVector{<:Number}, θ::Real, T::BallTree, centers_of_mass)
    h = T.hyper_spheres[index]
    if isleaf(T.tree_data.n_internal_nodes, index) # do direct computation
        y_ind = get_leaf_range(T.tree_data, index)
        y_i, w_i = @views y[T.indices[y_ind]], w[T.indices[y_ind]]

        elty = promote_type(eltype(x), eltype(eltype(y_i)), eltype(w_i))
        val = zero(elty)
        @inbounds @simd for j in eachindex(y_i)
          val += k(x, y_i[j]) * w_i[j]
        end
        return val

    elseif h.r < θ * euclidean(x, centers_of_mass[index]) # compress

        return 0

    else # recurse

        l = bh_recursion_no_far_field(getleft(index), k, x, y, w, sums_w, θ, T, centers_of_mass) # IDEA: these two are spawnable
        r = bh_recursion_no_far_field(getright(index), k, x, y, w, sums_w, θ, T, centers_of_mass)
        return l + r
    end
end

verbose = false
# IDEA: loop over element types, loop over kernels
@testset "barneshut" begin
    n = 1024
    d = 2
    rand_sign = rand(n) .* rand((-1, 1), n)
    weight_vectors = [ones(n), rand(n), rand_sign, randn(n)]
    weight_names = ["ones", "rand", "rand + sign", "randn"]

    X = randn(d, n)
    x = vector_of_static_vectors(X)
    # k = (x, y) -> inv(1 + sum(abs2, x-y))
    k = (x, y) -> CovarianceFunctions.Cauchy()(x, y)
    # k = (x, y) -> CovarianceFunctions.Exp()(x, y) # making it anonymous to test independence to kernel types
    K = gramian(k, x)
    # k = CovarianceFunctions.EQ()
    # k = function (x, y)
    #         r = sum(abs2, x-y)
    #         iszero(r) ? 0 : inv(r^2)
    #     end

    # test factorization syntax
    # leafsize = 16 # could also loop over this
    D = 1e-2
    F = BarnesHutFactorization(k, x, x, D; θ = 1/8, leafsize = 16)

    @testset "weight vector $name" for (w, name) in zip(weight_vectors, weight_names)
        # display(sum(eigvals(K + D*I) .> 1e-6))

        b = K*w + D*w
        b_bh = F*w
        @test isapprox(b, b_bh, rtol = 1e-3)
        w_bh = \(F, b_bh, maxiter = 128, verbose = false)
        @test isapprox(b_bh, F*w_bh, rtol = 1e-3)
        # @test isapprox(w, w_bh, rtol = 1e-3) # usually 1e-2 and less robust to smaller diagonal
        if verbose
            println("relative errors")
            println(norm(b-b_bh) / norm(b))
            println(norm(b_bh - F*w_bh) / norm(b_bh))
            println(norm(w - w_bh) / norm(w))
        end
        θ = 0
        b_bh = zero(b)
        barneshut!(b_bh, F, w, 1, 0, θ)
        @test b ≈ b_bh # testing that barnes hut is exact with θ = 0

        nexp = 16
        err = zeros(nexp)
        err_no_split = zeros(nexp)
        err_hyp = zeros(nexp)
        err_hyp_no_split = zeros(nexp)
        err_nff = zeros(nexp)
        theta_array = range(1e-1, 1, length = nexp)
        for (i, θ) in enumerate(theta_array)
            barneshut!(b_bh, F, w, 1, 0, θ, split = true, use_com = true)
            err[i] = norm(b - b_bh)

            θ_no_split = θ # just testing out how accuracy compares against splitting barnes hut
            barneshut!(b_bh, F, w, 1, 0, θ_no_split, split = false, use_com = true)
            err_no_split[i] = norm(b - b_bh)

            barneshut!(b_bh, F, w, 1, 0, θ, split = true, use_com = false) # use centers of hyperspheres
            err_hyp[i] = norm(b - b_bh)

            θ_no_split = θ
            barneshut!(b_bh, F, w, 1, 0, θ_no_split, split = false, use_com = false) # use centers of hyperspheres
            err_hyp_no_split[i] = norm(b - b_bh)

            barneshut_no_far_field!(b_bh, F, w, 1, 0, θ) # compare against pseudo barnes hut where far field = 0
            err_nff[i] = norm(b - b_bh)
        end

        rel_err = err / norm(b)
        @test all(<(1e-1), rel_err[1:findlast(<(0.5), theta_array)])
        @test all(<(1e-2), rel_err[1:findlast(<(0.2), theta_array)])

        # # plotting code
        # using Plots
        # plotly()
        #
        # rel_err_nff = err_nff / norm(b)
        # rel_err_no_split = err_no_split / norm(b)
        # rel_err_hyp = err_hyp / norm(b)
        # rel_err_hyp_no_split = err_hyp_no_split / norm(b)
        #
        # plot(theta_array, rel_err, yscale = :log10, label = "barneshut", ylabel = "relative error", xlabel = "θ")
        # plot!(theta_array, rel_err_no_split, yscale = :log10, label = "no split")
        # plot!(theta_array, rel_err_hyp, yscale = :log10, label = "hyp") # hyp meaning using centers of hyper balls
        # plot!(theta_array, rel_err_hyp_no_split, yscale = :log10, label = "hyp no split")
        # plot!(theta_array, rel_err_nff, yscale = :log10, label = "sparse")
        # gui()



    end # testset weight vectors
end # testset

end # module
