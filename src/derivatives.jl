using TaylorSeries
# computes all available derivatives, including value (0th order derivative)
derivatives(t::Taylor1) = derivatives!(similar(t.coeffs), t)
function derivatives!(d::AbstractVector, t::Taylor1)
    c, n = t.coeffs, t.order
    @. d = c * factorial(0:n)
end
# computes all derivatives up to mth order (including value)
function derivatives(f, x::Real, m::Int)
    d = zeros(eltype(x), m+1)
    derivatives!(d, f, x)
end
function derivatives(f, x::AbstractVector{<:Real}, m::Int)
    d = zeros(eltype(x), m+1, length(x))
    for (i, xi) in enumerate(x)
        d[:, i] .= derivatives(f, xi, m)
    end
    return d
end

function derivatives!(d::AbstractVector, f, x::Real)
    m = length(d) - 1
    @. d = 0
    d[1] = x
    d[2] = 1
    x = Taylor1(d, m) # mth order taylor series
    fx = f(x) # TODO: in place?
    derivatives!(d, fx) # overwrites coeffs vector with derivatives
end

# IDEA: could specialize for popular kernel types
# function derivatives!(d::AbstractVector, f::CovarianceFunctions.Exp, r::Real)
#     m = length(d)-1
#     @. d = f(r)
# end
