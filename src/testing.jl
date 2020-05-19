include("ski.jl")
using BenchmarkTools

# Testing stuff
struct PointGenerator
    f
end

function (f::PointGenerator)(n, bound)
    vect = rand(Uniform((-1 * bound), bound), n)
    return (map(x->(x, f.f(x)), vect))
end

# this function lifted from https://stackoverflow.com/questions/36367482/unzip-an-array-of-tuples-in-julia
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

expTest = PointGenerator(x->exp(x))
sinTest = PointGenerator(x->sin(x))
cossqTest = PointGenerator(x->(cos(x))^2)
contrivedFn1 = PointGenerator(x->(exp(cos(x))^2))
square = PointGenerator(x->x^2)
cube = PointGenerator(x->x^3)
natLog = PointGenerator(x->log(x))

tenPointsExpx, tenPointsExpy = unzip(expTest(10, 1000))
hundredPointsExpx, hundredPointsExpy = unzip(expTest(100, 1000))
thousandPointsExpx, thousandPointsExpy = unzip(expTest(1000, 1000))
tenThousandPointExpx, tenThousandPointExpy = unzip(expTest(10000, 1000))

tenPointsSinx, tenPointsSiny = unzip(sinTest(10, 1000))
hundredPointsSinx, hundredPointsSiny = unzip(sinTest(100, 1000))
thousandPointsSinx, thousandPointsSiny = unzip(sinTest(1000, 1000))
tenThousandPointSinx, tenThousandPointSiny = unzip(sinTest(10000, 1000))

tenPointsContrivedx, tenPointsContrivedy = unzip(contrivedFn1(10, 1000))
hundredPointsContrivedx, hundredPointsContrivedy = unzip(contrivedFn1(100, 1000))
thousandPointsContrivedx, thousandPointsContrivedy = unzip(contrivedFn1(1000, 1000))
tenThousandPointContrivedx, tenThousandPointContrivedy = unzip(contrivedFn1(10000, 1000))

expKern = Kernel.Exponential()

nonSKIExpKernExpTen = Kernel.Gramian(expKern, tenPointsExpx, tenPointsExpx)
nonSKIExpKernExpHundred = Kernel.Gramian(expKern, hundredPointsExpx, hundredPointsExpx)
nonSKIExpKernExpThousand = Kernel.Gramian(expKern, thousandPointsExpx, thousandPointsExpx)
nonSKIExpKernExpTenThousand = Kernel.Gramian(expKern, tenThousandPointExpx, tenThousandPointExpx)

nonSKIExpKernSinTen = Kernel.Gramian(expKern, tenPointsSinx, tenPointsSinx)
nonSKIExpKernSinHundred = Kernel.Gramian(expKern, hundredPointsSinx, hundredPointsSinx)
nonSKIExpKernSinThousand = Kernel.Gramian(expKern, thousandPointsSinx, thousandPointsSinx)
nonSKIExpKernSinTenThousand = Kernel.Gramian(expKern, tenThousandPointSinx, tenThousandPointSinx)

nonSKIExpKernContrivedTen = Kernel.Gramian(expKern, tenPointsContrivedx, tenPointsContrivedx)
nonSKIExpKernContrivedHundred = Kernel.Gramian(expKern, hundredPointsContrivedx, hundredPointsContrivedx)
nonSKIExpKernContrivedThousand = Kernel.Gramian(expKern, thousandPointsContrivedx, thousandPointsContrivedx)
nonSKIExpKernContrivedTenThousand = Kernel.Gramian(expKern, tenThousandPointContrivedx, tenThousandPointContrivedx)

suite = BenchmarkTools.BenchmarkGroup()
suite["expKernExpDatNonSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernExpDatNonSki"]["ten"] = @benchmarkable Matrix($nonSKIExpKernExpTen)
suite["expKernExpDatNonSki"]["hundred"] = @benchmarkable Matrix($nonSKIExpKernExpHundred)
suite["expKernExpDatNonSki"]["thousand"] = @benchmarkable Matrix($nonSKIExpKernExpThousand)
suite["expKernExpDatNonSki"]["ten thousand"] = @benchmarkable Matrix($nonSKIExpKernExpTenThousand)

suite["expKernSinDatNonSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernSinDatNonSki"]["ten"] = @benchmarkable Matrix($nonSKIExpKernSinTen)
suite["expKernSinDatNonSki"]["hundred"] = @benchmarkable Matrix($nonSKIExpKernSinHundred)
suite["expKernSinDatNonSki"]["thousand"] = @benchmarkable Matrix($nonSKIExpKernSinThousand)
suite["expKernSinDatNonSki"]["ten thousand"] = @benchmarkable Matrix($nonSKIExpKernSinTenThousand)

suite["expKernContrivedDatNonSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernContrivedDatNonSki"]["ten"] = @benchmarkable Matrix($nonSKIExpKernContrivedTen)
suite["expKernContrivedDatNonSki"]["hundred"] = @benchmarkable Matrix($nonSKIExpKernContrivedHundred)
suite["expKernContrivedDatNonSki"]["thousand"] = @benchmarkable Matrix($nonSKIExpKernContrivedThousand)
suite["expKernContrivedDatNonSki"]["ten thousand"] = @benchmarkable Matrix($nonSKIExpKernContrivedTenThousand)

suite["expKernExpDatSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernExpDatSki"]["ten"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenPointsExpx), 100)
suite["expKernExpDatSki"]["hundred"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(hundredPointsExpx), 1000)
suite["expKernExpDatSki"]["thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(thousandPointsExpx), 10000)
suite["expKernExpDatSki"]["ten thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenThousandPointExpx), 100000)

suite["expKernSinDatSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernSinDatSki"]["ten"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenPointsSinx), 100)
suite["expKernSinDatSki"]["hundred"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(hundredPointsSinx), 1000)
suite["expKernSinDatSki"]["thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(thousandPointsSinx), 10000)
suite["expKernSinDatSki"]["ten thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenThousandPointSinx), 100000)

suite["expKernContrivedDatSki"] = BenchmarkTools.BenchmarkGroup()
suite["expKernContrivedDatSki"]["ten"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenPointsContrivedx), 100)
suite["expKernContrivedDatSki"]["hundred"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(hundredPointsContrivedx), 1000)
suite["expKernContrivedDatSki"]["thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(thousandPointsContrivedx), 10000)
suite["expKernContrivedDatSki"]["ten thousand"] = @benchmarkable structured_kernel_interpolant(expKern, $sort(tenThousandPointContrivedx), 100000)


BenchmarkTools.run(suite)



