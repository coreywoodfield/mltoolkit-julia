
# exports for MLToolkit
export Matrix, Row, columns, rows, attributename, attributevalue, valuecount
export columnmean, columnmaximum, columnminimum, mostcommonvalue, shuffle!
export getrows, Split, splitmatrix

const numbertypes = Set(["REAL", "CONTINUOUS", "INTEGER"])
const MISSING = Inf
const Row = Vector{Float64}

"""
    Matrix <: AbstractMatrix{Float64}

2-d matrix containing arff data. To read a matrix in from an ARFF file, use
`loadarff(arff)`. To copy part of an existing matrix, use `copymatrix(...)`.
To initialize an empty matrix with a certain size, use `Matrix(rows, columns)`

You can iterate over the rows using a standard for loop:
```julia
for row in matrix
	...
end
```

Rows and individual elements may be accessed or set using index notation:
```julia
row1 = m[1]
element5 = m[1,5]
m[1,5] = 1.0
```
"""
struct Matrix <: AbstractMatrix{Float64}
	data::Vector{Row}
	attr_name::Vector{AbstractString}
	str_to_enum::Vector{Dict{AbstractString,Integer}}
	enum_to_str::Vector{Dict{Integer,AbstractString}}
	datasetname::AbstractString
end

# These functions allow the Matrix to act like a standard julia collection
# iterate over rows using `for row in m ... end`
# get a specific row using `m[1]` or a specific value using `m[1,5]`
Base.start(m::Matrix) = start(m.data)
Base.next(m::Matrix, state) = next(m.data, state)
Base.done(m::Matrix, state) = done(m.data, state)
# Base.eltype(::Type{Matrix}) = Vector{Float64}
Base.length(m::Matrix) = length(m.data)
Base.size(m::Matrix) = (rows(m), columns(m))
Base.getindex(m::Matrix, i::Int) = m.data[i]
Base.getindex(m::Matrix, i::Vararg{Int, 2}) = m.data[i[1]][i[2]]
Base.setindex!(m::Matrix, v::Row, i::Int) = m.data[i] = v
Base.setindex!(m::Matrix, v::Float64, i::Vararg{Int, 2}) = m.data[i[1]][i[2]] = v

const rows = Base.length
columns(m::Matrix) = length(m.attr_name)
attributename(m::Matrix, col::Integer) = m.attr_name[col]
setattributename(m::Matrix, col::Integer, name::AbstractString) = m.attr_name[col] = name
attributevalue(m::Matrix, col::Integer, value::Integer) = m.enum_to_str[col][value]
valuecount(m::Matrix, col::Integer) = length(m.enum_to_str[col])
columnmean(m::Matrix, col::Integer) = mean(m[:,col])
columnminimum(m::Matrix, col::Integer) = minimum(m[:,col])
columnmaximum(m::Matrix, col::Integer) = maximum(m[:,col])
mostcommonvalue(m::Matrix, col::Integer) = mostcommonvalue(m[:,col])
function mostcommonvalue(column)
	counts = Dict{Float64,Integer}()
	for value in column
		if value != MISSING
			counts[value] = get(counts, value, 0) + 1
		end
	end
	counts = map(reverse, counts)
	counts[maximum(keys(counts))]
end
shuffle!(m::Matrix) = permute!(m, randperm(rows(m)))
function shuffle!(m::Matrix, buddy::Matrix)
	perm = randperm(rows(m))
	permute!(m, perm)
	permute!(buddy, perm)
end

"""
    copymatrix(m, rows, columns)

Get the specified portion of Matrix `m` and returns it as a new matrix.
`rows` and `columns` should be a range or an array of indices, or
[any other index](https://docs.julialang.org/en/stable/manual/arrays/#man-supported-index-types-1)
supported by standard julia indexing.

This gets a view of the original matrix, and not a copy. If you modify this
matrix, the original matrix will also be modified.

!!! note

    The `Matrix` class is 1-indexed. Thus, the values for rows and columns should
    be between 1 and the number of rows or columns, inclusively.
"""
function copymatrix(m::Matrix, rows, columns)
	data = map(i->m.data[i][columns], rows)
	attr_name = m.attr_name[columns]
	str_to_enum = m.str_to_enum[columns]
	enum_to_str = m.enum_to_str[columns]
	Matrix(data, attr_name, str_to_enum, enum_to_str, m.datasetname)
end
copymatrix(m::Matrix, rows, columns::Int) = copymatrix(m, rows, range(columns, 1))

"""
    getrows(m, rows)

Get the specified rows from the matrix m, as a matrix. `rows` can be a range or an array of indices, or
[any other index](https://docs.julialang.org/en/stable/manual/arrays/#man-supported-index-types-1)
supported by standard julia indexing.
"""
getrows(m::Matrix, rows) = copymatrix(m, rows, 1:columns(m))

struct Split
	trainfeatures::Matrix
	trainlabels::Matrix
	validationfeatures::Matrix
	validationlabels::Matrix
end

"""
    split(features, labels, percenttest)

Split the given matrices into a training set and a validation set, where
`percenttest`% of the rows are put in the validation set and `1-percenttest`%
of the rows are put into the training set.
"""
function splitmatrix(features::Matrix, labels::Matrix, percenttest::AbstractFloat)
	shuffle!(features, labels)
	numrows = rows(features)
	trainrows = trunc(Int, (1 - percenttest) * numrows)
	trainfeatures = getrows(features, 1:trainrows)
	trainlabels = getrows(labels, 1:trainrows)
	validationfeatures = getrows(features, trainrows+1:numrows)
	validationlabels = getrows(labels, trainrows+1:numrows)
	Split(trainfeatures, trainlabels, validationfeatures, validationlabels)
end

function Matrix(rows::Integer, columns::Integer)
	data = map(_ -> zeros(columns), 1:rows)
	attr_name = fill("", columns)
	# kinda hacky, but whatever - creates a new (empty) dictionary for each column
	str_to_enum = map(_ -> Dict{AbstractString,Integer}(), 1:columns)
	enum_to_str = map(_ -> Dict{Integer,AbstractString}(), 1:columns)
	Matrix(data, attr_name, str_to_enum, enum_to_str, "")
end

function loadarff(filename::AbstractString)::Matrix
	io = open(filename)
	# skip empty lines and comments at the beginning
	skipchars(io, isspace; linecomment='%')
	# initialize variables
	attr_name = Vector{AbstractString}()
	str_to_enum = Vector{Dict{AbstractString,Integer}}()
	enum_to_str = Vector{Dict{Integer,AbstractString}}()
	datasetname = ""
	# read attributes - break when you get to the data
	while true
		line = readline(io)
		(length(line) == 0 || line[1] == '%') && continue
		upper = uppercase(line)
		if startswith(upper, "@RELATION")
			# Everything after Relation is the datasetname
			datasetname = split(line, ' '; limit=2)[2]
		elseif startswith(upper, "@ATTRIBUTE")
			# Attribute should have three distinct parts - @Attribute, name, and type
			# e.g. "@attribute	'handicapped-infants'	{ 'n', 'y'}"
			# or "@ATTRIBUTE	sepallength	Continuous"
			attribute = split(line, r"\s+"; limit=3, keep=false)
			ste = Dict{AbstractString,Integer}()
			ets = Dict{Integer,AbstractString}()
			push!(attr_name, attribute[2])
			push!(str_to_enum, ste)
			push!(enum_to_str, ets)
			# If it's one of the number types, there's no need to do anything with it
			if uppercase(attribute[3]) ∉ numbertypes
				stripped = strip(attribute[3], ['{', '}', ' '])
				values = split(stripped, [' ', ',']; keep=false)
				foreach(values, Iterators.countfrom(0)) do val, i
					ste[val] = i
					ets[i] = val
				end
			end
		elseif startswith(upper, "@DATA")
			break
		end
	end
	data = Vector{Row}()
	while !eof(io)
		line = readline(io)
		(length(line) == 0 || line[1] == '%') && continue
		line = map(strip, split(line, ','))
		row = map(getfloatvalue, line, str_to_enum)
		push!(data, row)
	end
	close(io)
	Matrix(data, attr_name, str_to_enum, enum_to_str, datasetname)
end

function getfloatvalue(value::AbstractString, dict::Dict{AbstractString,Integer})::Float64
	value = strip(value)
	if length(dict) == 0
		parse(Float64, value)
	elseif haskey(dict, value)
		dict[value]
	elseif value == "?"
		MISSING
	else
		error("Error parsing value: $value with dict: $dict")
	end
end

function normalize(m::Matrix)
	# get a list of mins and maxes for each column
	extr = vec(extrema(m, 1))
	normalize(m, extr)
	extr
end

"""
    normalize(m[, extrema])

Normalizes the matrix `m` so that all columns have values between 0 and 1.

If `extrema` is included, the matrix is normalized as though the maximum and
minimum value of each column were the values included in extrema.
This allows for two matrices to be normalized using the same ranges.

If `extrema` is not included, this method returns a list of extrema that can
then be used to normalize another matrix.
"""
function normalize(m::Matrix, extrema::Vector{Tuple{Float64,Float64}})
	cols = columns(m)
	values = map(c -> valuecount(m, c), 1:cols)
	for row in m
		for (i, (value, count, (min, max))) in enumerate(zip(row, values, extrema))
			if count == 0 && value != MISSING
				row[i] = (value - min) / (max - min)
			end
		end
	end
end

function Base.show(io::IO, m::Matrix)
	println(io, "@RELATION ", m.datasetname)
	for (name, enum_to_str) in zip(m.attr_name, m.enum_to_str)
		println(io, "@ATTRIBUTE ", name, " ", begin
			valcount = length(enum_to_str)
			if valcount == 0
				"CONTINUOUS"
			else
				"{", join(values(enum_to_str), ", "), "}"
			end
		end...)
	end
	println(io, "@DATA")
	for row in m.data
		mapped = map(row, m.enum_to_str) do val, map
			length(map) == 0 ? val : map[val]
		end
		join(io, mapped, ", ")
		println()
	end
end
