############################ Gradient Algebra ##################################
################################# Sum ##########################################
# allocates space for gradient kernel evaluation but does not evaluate
# separation from evaluation useful for ValueGradientKernel
function allocate_gradient_kernel(k::Sum, x, y, ::GenericInput)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    LazyMatrixSum(H...)
end

function gradient_kernel!(K::LazyMatrixSum, k::Sum, x::AbstractVector, y::AbstractVector, ::GenericInput)
    for (h, H) in zip(k.args, K.args)
        gradient_kernel!(H, h, x, y, input_trait(h))
    end
    return K
end

################################ Product #######################################
# for product kernel with generic input
function allocate_gradient_kernel(k::Product, x, y, ::GenericInput)
    d, r = length(x), length(k.args)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    T = typeof(k(x, y))
    A = LazyMatrixSum(
                (LazyMatrixProduct(Diagonal(zeros(T, d)), h) for h in H)...
                )
    U = zeros(T, (d, r))
    V = zeros(T, (d, r))
    C = Woodbury(I(r), ones(r), ones(r)', -1)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::Product, x::AbstractVector, y::AbstractVector, ::GenericInput = input_trait(k))
    A = W.A # this is a LazyMatrixSum of LazyMatrixProducts
    ForwardDiff.jacobian!(W.U, z->k(z, y), x)
    ForwardDiff.jacobian!(W.V', z->k(x, z), y)
    k_j = [h(x, y) for h in k.args]
    k_j .= prod(k_j) ./ k_j
    r = length(k.args)
    for i in 1:r
        h, H = k.args[i], A.args[i]
        D = H.args[1]
        @. D.diag = k_j
        gradient_kernel!(H.args[2], h, x, y, input_trait(h))
    end
    return W
end

############################# Separable Product ################################
# for product kernel with generic input
function allocate_gradient_kernel(k::SeparableProduct, x::AbstractVector{<:Number},
                                  y::AbstractVector{<:Number}, ::GenericInput)
    d = length(x)
    H = (allocate_gradient_kernel(h, x, y, input_trait(h)) for h in k.args)
    T = typeof(k(x, y))
    A = LazyMatrixProduct(Diagonal(zeros(T, d)), Diagonal(zeros(T, d)))
    U = Diagonal(zeros(T, d))
    V = Diagonal(zeros(T, d))
    C = Woodbury(I(r), ones(r), ones(r)', -1)
    Woodbury(A, U, C, V)
end

function gradient_kernel!(W::Woodbury, k::SeparableProduct, x::AbstractVector, y::AbstractVector, ::GenericInput = input_trait(k))
    A = W.A # this is a LazyMatrixProducts of Diagonals
    D, H = A.args # first is scaling matrix by leave_one_out_products, second is diagonal of derivative kernels
    for (i, ki) in enumerate(k.args)
        xi, yi = x[i], y[i]
        D.diag[i, i] = ki(xi, yi)
        W.U[i, i] = ForwardDiff.derivative(z->ki(z, yi), xi)
        W.V[i, i] = ForwardDiff.derivative(z->ki(xi, z), yi)
        H[i, i] = DerivativeKernel(ki)(xi, yi)
    end
    leave_one_out_products!(D.diag)
    return W
end

############################# Separable Sum ####################################
# IDEA: implement block separable with x::AbstractVecOfVec
function allocate_gradient_kernel(k::SeparableSum, x::AbstractVector{<:Number},
                                  y::AbstractVector{<:Number}, ::GenericInput)
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    D = Diagonal(d)
end

function gradient_kernel!(D::Diagonal, k::SeparableSum, x::AbstractVector{<:Number},
                          y::AbstractVector{<:Number}, ::GenericInput)
    for (i, ki) in enumerate(k.args)
        D[i, i] = DerivativeKernel(ki)(x[i], y[i])
    end
    return D
end

############################## Input Transformations ###########################
# can be most efficiently represented by factoring out the Jacobian w.r.t. input warping
function gramian(G::GradientKernel{<:Real, <:Warped},  x::AbstractVector, y::AbstractVector)
    W = G.k
    U(x) = BlockFactorization(Diagonal([ForwardDiff.jacobian(W.u, xi) for xi in x]))
    k = GradientKernel(W.k)
    LazyMatrixProduct(U(x)', gramian(k, x, y), U(y))
end

function gramian(G::GradientKernel{<:Real, <:ScaledInputKernel},  x::AbstractVector, y::AbstractVector)
    n, m = length(x), length(y)
    S = G.k
    Ux = kronecker(I(n), S.U)
    Uy = n == m ? Ux : kronecker(I(m), S.U)
    k = GradientKernel(S.k)
    LazyMatrixProduct(Ux', gramian(k, x, y), Uy)
end

function gramian(G::GradientKernel{<:Real, <:Lengthscale}, x::AbstractVector, y::AbstractVector)
    n, m = length(x), length(y)
    L = G.k
    Ux = Diagonal(fill(L.l, d*n)) # IDEA: Fill for lazy uniform array
    Uy = n == m ? Ux : Diagonal(fill(L.l, d*m))
    k = GradientKernel(L.k)
    LazyMatrixProduct(Ux', gramian(k, x, y), Uy)
end

############################### VerticalRescaling ##############################
# gradient element can be expressed with WoodburyIdentity and LazyMatrixProduct
function allocate_gradient_kernel(k::VerticalRescaling, x, y, ::GenericInput = GenericInput())
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    A = LazyMatrixProduct(Diagonal(fill(f(x), d)), H, Diagonal(fill(f(y), d)))
    U = zeros(d, 2)
    V = zeros(d, 2)
    C = zeros(2, 2)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::VerticalRescaling, x, y, ::GenericInput = GenericInput())
    f, h, A = k.f, k.k, W.A
    fx, fy = f(x), f(y)
    @. A.args[1].diag = fx
    H = A.args[2] # LazyMatrixProduct: first and third are the diagonal scaling matrices, second is the gradient_kernel_matrix of h
    @. A.args[3].diag = fy
    gradient_kernel!(H, h, x, y, input_trait(h))
    ForwardDiff.gradient!(@view(W.U[:, 1]), f, x)
    ForwardDiff.gradient!(@view(W.U[:, 2]), z->h(z, y), x)
    ForwardDiff.gradient!(@view(W.V[1, :]), f, y)
    ForwardDiff.gradient!(@view(W.V[2, :]), z->h(x, z), y)
    W.C[1, 1] = h(x, y)
    W.C[1, 2] = fy
    W.C[2, 1] = fx
    return W
end

############################ Scalar Chain Rule #################################
# generic implementation of scalar chain rule, does not require input kernel to have a basic input type
# gradient element can be expressed with WoodburyIdentity and LazyMatrixProduct
function allocate_gradient_kernel(k::Chained, x, y, ::GenericInput)
    f, h, d = k.f, k.k, length(x)
    H = allocate_gradient_kernel(h, x, y, input_trait(h))
    A = LazyMatrixProduct(Diagonal(fill(f(h(x, y)), d)), H)
    U = zeros(d, 1)
    V = zeros(d, 1)
    C = zeros(1, 1)
    Woodbury(A, U, C, V')
end

function gradient_kernel!(W::Woodbury, k::Chained, x, y, ::GenericInput)
    f, h, A = k.f, k.k, W.A
    f1, f2 = derivative_laplacian(f, h(x, y))
    @. A.args[1].diag = f1
    H = A.args[2] # LazyMatrixProduct: first argument is diagonal scaling, second is the gradient_kernel_matrix of h
    gradient_kernel!(H, h, x, y, input_trait(h))
    ForwardDiff.gradient!(@view(W.U[:]), z->h(z, y), x)
    ForwardDiff.gradient!(@view(W.V[1, :]), z->h(x, z), y)
    @. W.C = f2
    return W
end
