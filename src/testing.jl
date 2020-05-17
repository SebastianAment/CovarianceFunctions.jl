
using Kernel
using Distributions
# Testing stuff


struct PointGenerator
    f
end

function (f::PointGenerator)(n, bound)
    vect = rand(Uniform((-1 * bound), bound), n)
    return (map(x->(x, f.f(x)), vect))
end

expTest = PointGenerator(x->exp(x))
sinTest = PointGenerator(x->sin(x))
cossqTest = PointGenerator(x->(cos(x))^2)
contrivedFn1 = PointGenerator(x->(exp(cos(x))^2))
square = PointGenerator(x->x^2)
cube = PointGenerator(x->x^3)
natLog = PointGenerator(x->log(x))