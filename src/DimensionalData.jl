module DimensionalData

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DimensionalData

# Standard lib
using Dates,
      LinearAlgebra,
      Random,
      Statistics,
      SparseArrays

using Base.Broadcast: Broadcasted, BroadcastStyle, DefaultArrayStyle, AbstractArrayStyle,
      Unknown

using Base: tail, OneTo, @propagate_inbounds
      
# Ecosystem
import Adapt, 
       ArrayInterface,
       ConstructionBase, 
       RecipesBase,
       Tables

using RecipesBase: @recipe



export Dimension, IndependentDim, DependentDim, XDim, YDim, ZDim, TimeDim,
       X, Y, Z, Ti, ParametricDimension, Dim, AnonDim, Coord

export Selector, At, Between, Contains, Near, Where

export Locus, Center, Start, End, AutoLocus

export Order, Ordered, Unordered, UnknownOrder, AutoOrder

export IndexOrder, ArrayOrder, Relation,
       ForwardIndex, ReverseIndex, UnorderedIndex,
       ForwardArray, ReverseArray, ForwardRelation, ReverseRelation

export Sampling, Points, Intervals

export Span, Regular, Irregular, Explicit, AutoSpan

export Mode, IndexMode, Auto, AutoMode, NoIndex

export Aligned, AbstractSampled, Sampled, AbstractCategorical, Categorical

export Unaligned, Transformed

export AbstractDimArray, DimArray, AbstractDimensionalArray, DimensionalArray

export AbstractDimTable, DimTable

export AbstractDimStack, DimStack

export DimIndices, DimKeys

export AbstractMetadata, Metadata, NoMetadata

export AbstractName, Name, NoName

export data, dims, refdims, mode, metadata, name, label, units,
       val, index, order, sampling, span, bounds, locus, <|

export dimnum, hasdim, otherdims, commondims, setdims, swapdims, sortdims,
       set, rebuild, reorder, modify, dimwise, dimwise!

export order, indexorder, arrayorder, relation

export @dim

const DD = DimensionalData

include("interface.jl")
include("mode.jl")
include("metadata.jl")
include("name.jl")
include("identify.jl")
include("dimension.jl")
include("array.jl")
include("stack.jl")
include("tables.jl")
include("selector.jl")
include("dimindices.jl")
include("indexing.jl")
include("primitives.jl")
include("broadcast.jl")
include("methods.jl")
include("matmul.jl")
include("coord.jl")
include("set.jl")
include("utils.jl")
include("plotrecipes.jl")
include("show.jl")

# For compat with old versions
const AbstractDimensionalArray = AbstractDimArray
const DimensionalArray = DimArray

const AbstractDimDataset = AbstractDimStack
const DimDataset = DimStack

function _precompile()
    precompile(DimArray, (Array{Int,1}, typeof(X())))
    precompile(DimArray, (Array{Int,2}, Tuple{typeof(X()), typeof(Y())}))
    precompile(DimArray, (Array{Float64,1}, typeof(X())))
    precompile(DimArray, (Array{Float64,2}, Tuple{typeof(X()), typeof(Y())}))
    precompile(DimArray, (Array{Float64,3}, Tuple{typeof(X()), typeof(Y()), typeof(Z())}))
end

_precompile()

end
