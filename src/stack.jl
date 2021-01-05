"""
Supertype for dimensional stacks.

These have multiple layers of data, but share dimensions.
"""
abstract type AbstractDimStack{L,D} end

data(s::AbstractDimStack) = s.data
dims(s::AbstractDimStack) = s.dims
refdims(s::AbstractDimStack) = s.refdims
layerdims(s::AbstractDimStack) = s.layerdims
metadata(s::AbstractDimStack) = s.metadata
metadata(s::AbstractDimStack, key::Symbol) = _metadata(data(s), key)

_metadata(m::NamedTuple, key) = m[key]
_metadata(m, key) = m

_basedim(x) = _basedim(dims(x))
_basedim(ds::Tuple) = map(_basedim, ds)
_basedim(d::Dimension) = basetypeof(d)()

Base.keys(s::AbstractDimStack) = keys(data(s))
Base.values(s::AbstractDimStack) = values(dimarrays(s))
Base.first(s::AbstractDimStack) = first(dimarrays(s))
# Only compare data and dim - metadata and refdims can be different
Base.:(==)(s1::AbstractDimStack, s2::AbstractDimStack) =
    data(s1) == data(s2) && dims(s1) == dims(s2) && layerdims(s1) == layerdims(s2)

rebuild(s::AbstractDimStack, data, dims=dims(s), refdims=refdims(s), 
        layerdims=layerdims(s), metadata=metadata(s)) =
    basetypeof(s)(data, dims, refdims, layerdims, metadata)

function rebuildsliced(s::AbstractDimStack, data, I) 
    layerdims = map(_basedim, data)
    rebuild(s, map(parent, data), slicedims(s, I)..., layerdims)
end

function dimarrays(s::AbstractDimStack{<:NamedTuple{Keys}}) where Keys
    map(Keys, values(data(s)), values(layerdims(s))) do k, A, ld
        DimArray(A, dims(s, ld), refdims(s), k, NoMetadata())
    end |> NamedTuple{Keys}
end


Adapt.adapt_structure(to, s::AbstractDimStack) = map(A -> adapt(to, A), s)

# Dipatch on Tuple of Dimension, and map
for func in (:index, :mode, :metadata, :sampling, :span, :bounds, :locus, :order)
    @eval ($func)(s::AbstractDimStack, args...) = ($func)(dims(s), args...)
end

"""
    Base.map(f, s::AbstractDimStack)

Apply functrion `f` to each layer of the stack `s`, and rebuild it.

If `f` returns `DimArray`s the result will be another `DimStack`.
Other values will be returned in a `NamedTuple`.
"""
Base.map(f, s::AbstractDimStack) = _maybestack(map(f, dimarrays(s)))

_maybestack(As::NamedTuple{<:Any,<:Tuple{Vararg{<:AbstractDimArray}}}) = DimStack(As)
_maybestack(x::NamedTuple) = x

# Array methods

# Methods with no arguments that return a DimStack
for (mod, fnames) in
    (:Base => (:inv, :adjoint, :transpose), :LinearAlgebra => (:Transpose,))
    for fname in fnames
        @eval ($mod.$fname)(s::AbstractDimStack) = map(A -> ($mod.$fname)(A), s)
    end
end

# Methods with an argument that return a DimStack
for fname in (:rotl90, :rotr90, :rot180, :PermutedDimsArray, :permutedims)
    @eval (Base.$fname)(s::AbstractDimStack, args...) =
        map(A -> (Base.$fname)(A, args...), s)
end

# Methods with keyword arguments that return a DimStack
for (mod, fnames) in
    (:Base => (:sum, :prod, :maximum, :minimum, :extrema, :dropdims),
     :Statistics => (:cor, :cov, :mean, :median, :std, :var))
    for fname in fnames
        @eval ($mod.$fname)(s::AbstractDimStack; kw...) =
            _maybestack(map(A -> ($mod.$fname)(A; kw...), dimarrays(s)))
    end
end

# Methods that take a function
for (mod, fnames) in (:Base => (:reduce, :sum, :prod, :maximum, :minimum, :extrema),
                      :Statistics => (:mean,))
    for fname in fnames
        _fname = Symbol(:_, fname)
        @eval begin
            ($mod.$fname)(f::Function, s::AbstractDimStack; dims=Colon()) =
                ($_fname)(f, s, dims)
            # Colon returns a NamedTuple
            ($_fname)(f::Function, s::AbstractDimStack, dims::Colon) =
                map(A -> ($mod.$fname)(f, A), data(s))
            # Otherwise return a DimStack
            ($_fname)(f::Function, s::AbstractDimStack, dims) =
                map(A -> ($mod.$fname)(f, A; dims=dims), s)
        end
    end
end


"""
    DimStack(data::AbstractDimArray...)
    DimStack(data::Tuple{Vararg{<:AbstractDimArray}})
    DimStack(data::NamedTuple{Keys,Vararg{<:AbstractDimArray}})
    DimStack(data::NamedTuple, dims::DimTuple; metadata=nothing)

DimStack holds multiple objects sharing some dimensions, in a `NamedTuple`.
Indexing operates as for [`AbstractDimArray`](@ref), except it occurs for all
data layers of the stack simulataneously. Layer objects can hold values of any type.

DimStack can be constructed from multiple `AbstractDimArray` or a `NamedTuple`
of `AbstractArray` and a matching `dims` `Tuple`. If `AbstractDimArray`s have
the same name they will be given the name `:layer1`, substitiuting the
layer number for `1`.

`getindex` with `Int` or `Dimension`s or `Selector`s that resolve to `Int` will
return a `NamedTuple` of values from each layer in the stack. This has very good
performace, and usually takes less time than the sum of indexing each array
separately.

Indexing with a `Vector` or `Colon` will return another `DimStack` where
all data layers have been sliced.  `setindex!` must pass a `Tuple` or `NamedTuple` maching
the layers.

Most `Base` and `Statistics` methods that apply to `AbstractArray` can be used on
all layers of the stack simulataneously. The result is a `DimStack`, or
a `NamedTuple` if methods like `mean` are used without `dims` arguments, and
return a single non-array value.

## Example

```jldoctest
julia> using DimensionalData

julia> A = [1.0 2.0 3.0; 4.0 5.0 6.0];

julia> dimz = (X([:a, :b]), Y(10.0:10.0:30.0))
(X (type X): Symbol[a, b] (AutoMode), Y (type Y): 10.0:10.0:30.0 (AutoMode))

julia> da1 = DimArray(1A, dimz, :one);



julia> da2 = DimArray(2A, dimz, :two);



julia> da3 = DimArray(3A, dimz, :three);



julia> s = DimStack(da1, da2, da3)
DimStack{NamedTuple{(:one, :two, :three),Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}}},2,Tuple{X{Array{Symbol,1},Categorical{Unordered{ForwardRelation}},NoMetadata},Y{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Sampled{Ordered{ForwardIndex,ForwardArray,ForwardRelation},Regular{Float64},Points},NoMetadata}},Tuple{},NamedTuple{(:one, :two, :three),Tuple{Nothing,Nothing,Nothing}}}((one = [1.0 2.0 3.0; 4.0 5.0 6.0], two = [2.0 4.0 6.0; 8.0 10.0 12.0], three = [3.0 6.0 9.0; 12.0 15.0 18.0]), (X (type X): Symbol[a, b] (Categorical: Unordered), Y (type Y): 10.0:10.0:30.0 (Sampled: Ordered Regular Points)), (), (one = nothing, two = nothing, three = nothing))

julia> s[:b, 10.0]
(one = 4.0, two = 8.0, three = 12.0)

julia> s[X(:a)]
DimStack{NamedTuple{(:one, :two, :three),Tuple{DimArray{Float64,1,Tuple{Y{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Sampled{Ordered{ForwardIndex,ForwardArray,ForwardRelation},Regular{Float64},Points},NoMetadata}},Tuple{X{Symbol,Categorical{Unordered{ForwardRelation}},NoMetadata}},Array{Float64,1},Symbol,Nothing},DimArray{Float64,1,Tuple{Y{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Sampled{Ordered{ForwardIndex,ForwardArray,ForwardRelation},Regular{Float64},Points},NoMetadata}},Tuple{X{Symbol,Categorical{Unordered{ForwardRelation}},NoMetadata}},Array{Float64,1},Symbol,Nothing},DimArray{Float64,1,Tuple{Y{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Sampled{Ordered{ForwardIndex,ForwardArray,ForwardRelation},Regular{Float64},Points},NoMetadata}},Tuple{X{Symbol,Categorical{Unordered{ForwardRelation}},NoMetadata}},Array{Float64,1},Symbol,Nothing}}},1,Tuple{Y{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Sampled{Ordered{ForwardIndex,ForwardArray,ForwardRelation},Regular{Float64},Points},NoMetadata}},Tuple{X{Symbol,Categorical{Unordered{ForwardRelation}},NoMetadata}},NamedTuple{(:one, :two, :three),Tuple{Nothing,Nothing,Nothing}}}((one = DimArray (named one) with dimensions:
 Y (type Y): 10.0:10.0:30.0 (Sampled: Ordered Regular Points)
and referenced dimensions:
 X (type X): a (Categorical: Unordered)
and data: 3-element Array{Float64,1}
[1.0, 2.0, 3.0], two = DimArray (named two) with dimensions:
 Y (type Y): 10.0:10.0:30.0 (Sampled: Ordered Regular Points)
and referenced dimensions:
 X (type X): a (Categorical: Unordered)
and data: 3-element Array{Float64,1}
[2.0, 4.0, 6.0], three = DimArray (named three) with dimensions:
 Y (type Y): 10.0:10.0:30.0 (Sampled: Ordered Regular Points)
and referenced dimensions:
 X (type X): a (Categorical: Unordered)
and data: 3-element Array{Float64,1}
[3.0, 6.0, 9.0]), (Y (type Y): 10.0:10.0:30.0 (Sampled: Ordered Regular Points),), (X (type X): a (Categorical: Unordered),), (one = nothing, two = nothing, three = nothing))
```

"""
struct DimStack{L,D,R,LD,M} <: AbstractDimStack{L,D}
    data::L
    dims::D
    refdims::R
    layerdims::LD
    metadata::M
    function DimStack(data::L, dims::D, refdims::R, layerdims::LD, metadata::M) where {L,D,R,LD,M}
        new{L,D,R,LD,M}(data, dims, refdims, layerdims, metadata)
    end
end
DimStack(das::AbstractDimArray...; kwargs...) = DimStack(das; kwargs...)
function DimStack(das::Tuple{Vararg{<:AbstractDimArray}}; kwargs...)
    DimStack(NamedTuple{uniquekeys(das)}(das); kwargs...)
end
function DimStack(As::NamedTuple{<:Any,<:Tuple{Vararg{<:AbstractDimArray}}}; 
    refdims=(), metadata=map(metadata, As)
)
    data = map(parent, As)
    dims = combinedims(As...)
    layerdims = map(A -> _basedim(A), As)
    DimStack(data, dims, refdims, layerdims, metadata)
end
# Same sized arrays
function DimStack(data::NamedTuple, dims::DimTuple; refdims=(), metadata=nothing)
    all(map(d -> axes(d) == axes(first(data)), data)) || _stack_size_mismatch()
    layerdims = map(_ -> _basedim(dims), data)
    DimStack(data, formatdims(first(data), dims), refdims, layerdims, metadata)
end

@noinline _stack_size_mismatch() = throw(ArgumentError("Arrays must have identical axes. For mixed dimensions, use DimArrays`"))
