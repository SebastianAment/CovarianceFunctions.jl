function LinearAlgebra.givensAlgorithm(f::T, g::T) where T<:ForwardDiff.Dual # T<:Real?
    onepar = one(T)
    twopar = 2one(T)
    T0 = typeof(onepar) # dimensionless
    zeropar = T0(zero(T)) # must be dimensionless

    # need both dimensionful and dimensionless versions of these:
    safmn2 = LinearAlgebra.floatmin2(T0)
    safmn2u = LinearAlgebra.floatmin2(T)
    safmx2 = one(T)/safmn2
    safmx2u = oneunit(T)/safmn2

    if g == 0
        cs = onepar
        sn = zeropar
        r = f
    elseif f == 0
        cs = zeropar
        sn = onepar
        r = g
    else
        f1 = f
        g1 = g
        scalepar = max(abs(f1), abs(g1))
        if scalepar >= safmx2u
            count = 0
            while true
                count += 1
                f1 *= safmn2
                g1 *= safmn2
                scalepar = max(abs(f1), abs(g1))
                if scalepar < safmx2u break end
            end
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
            for i = 1:count
                r *= safmx2
            end
        elseif scalepar <= safmn2u
            count = 0
            while true
                count += 1
                f1 *= safmx2
                g1 *= safmx2
                scalepar = max(abs(f1), abs(g1))
                if scalepar > safmn2u break end
            end
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
            for i = 1:count
                r *= safmn2
            end
        else
            r = sqrt(f1*f1 + g1*g1)
            cs = f1/r
            sn = g1/r
        end
        if abs(f) > abs(g) && cs < 0
            cs = -cs
            sn = -sn
            r = -r
        end
    end
    return cs, sn, r
end
