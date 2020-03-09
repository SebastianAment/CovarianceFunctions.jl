# Kernels arising from phyiscal problems

# 2D Electromagnetic field
struct Electro{T} <: Isotropic{T} end
(::Electro{T})(r::T) where {T} = r ≈ 0 ? zero(T) : 1/(2π) * log(abs(r))
