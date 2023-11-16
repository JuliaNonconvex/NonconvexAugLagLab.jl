abstract type AbstractAggregation <: AbstractFunction end
getdim(::AbstractAggregation) = 1

struct NonNegSumOfSquares{T<:Real} <: AbstractAggregation
    c::T
end
(f::NonNegSumOfSquares)(x::AbstractVector) = f.c * sum(max.(0, x) .^ 2)

struct NonNegSumOfRoots{T<:Real} <: AbstractAggregation
    c::T
end
(f::NonNegSumOfRoots)(x::AbstractVector) = f.c * sum(sqrt.(max.(0, x)))

struct NonNegSum{T<:Real} <: AbstractAggregation
    c::T
end
(f::NonNegSum)(x::AbstractVector) = f.c * sum(max.(0, x))

struct WeightedSum{W<:AbstractVector} <: AbstractAggregation
    weights::W
end
(f::WeightedSum)(x::AbstractVector) = dot(f.weights, x)
