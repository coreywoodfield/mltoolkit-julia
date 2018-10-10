
const Row = Vector{Union{Float64,Missing}}

"""
    Matrix <: AbstractMatrix{Float64}

2-d matrix (typically) containing data from an arff file

To read a matrix in from an ARFF file, use
[`loadarff`](@ref). To copy part of an existing matrix, use [`copymatrix`](@ref).
To initialize an empty matrix with a certain size, use [`Matrix(rows, columns)`](@ref)

You can iterate over the rows using a standard for loop:
```julia
for row in matrix
	...
end
```

Rows and individual elements may be accessed or set using index notation:
```julia
row1 = matrix[1]
element5 = matrix[1,5]
matrix[1,5] = 1.0
```
"""
struct Matrix <: AbstractMatrix{Union{Float64,Missing}}
	rows::Vector{Row}
	attr_name::Vector{AbstractString}
	str_to_enum::Vector{Dict{AbstractString,Integer}}
	enum_to_str::Vector{Dict{Integer,AbstractString}}
	datasetname::AbstractString
end

# These functions allow the Matrix to act like a standard julia collection
# iterate over rows using `for row in m ... end`
# get a specific row using `m[1]` or a specific value using `m[1,5]`
Base.iterate(m::Matrix) = iterate(m.rows)
Base.iterate(m::Matrix, state) = iterate(m.rows, state)
Base.HasLength(::Type{Matrix}) = Base.HasShape{2}()
Base.length(m::Matrix) = length(m.rows)
Base.size(m::Matrix) = (rows(m), columns(m))
Base.firstindex(::Matrix) = 1
Base.lastindex(m::Matrix) = lastindex(m.rows)
Base.getindex(m::Matrix, i::Int) = m.rows[i]
Base.getindex(m::Matrix, i::Int, j::Int) = m.rows[i][j]
Base.setindex!(m::Matrix, v::Row, i::Int) = m.rows[i] = v
Base.setindex!(m::Matrix, v::Float64, i::Int, j::Int) = m.rows[i][j] = v

"    rows(matrix)"
const rows = Base.length
"    columns(matrix)"
columns(m::Matrix) = length(m.attr_name)
"    attributename(matrix, column)"
attributename(m::Matrix, col::Integer) = m.attr_name[col]
"    setattributename(matrix, column, name)"
setattributename(m::Matrix, col::Integer, name::AbstractString) = m.attr_name[col] = name
"""
    attributevalue(matrix, column, value)

Get the string representation of a value in a column
"""
attributevalue(m::Matrix, col::Integer, value) = m.enum_to_str[col][convert(Integer, value)]
"""
    valuecount(matrix, column)

If the column is a nominal feature, get the number of different valid values.
If the column is not nominal, 0.
"""
valuecount(m::Matrix, col::Integer) = length(m.enum_to_str[col])
function applytocolumn(f::Function, m::Matrix, col::Integer)
	column = collect(skipmissing(m[:, col]))
	if isempty(column)
		missing
	else
		f(column)
	end
end
"    columnmean(matrix, column)"
columnmean(m::Matrix, col::Integer) = applytocolumn(mean, m, col)
"    columnminimum(matrix, column)"
columnminimum(m::Matrix, col::Integer) = applytocolumn(minimum, m, col)
"    columnmaximum(matrix, column)"
columnmaximum(m::Matrix, col::Integer) = applytocolumn(maximum, m, col)
"    mostcommonvalue(matrix, column)"
mostcommonvalue(m::Matrix, col::Integer) = applytocolumn(mostcommonvalue, m, col)
function mostcommonvalue(column)
	counts = Dict{Float64,Integer}()
	for value in column
		counts[value] = get(counts, value, 0) + 1
	end
	partialsort!(collect(counts), 1, by=x->x[2], rev=true)[1]
end
"""
    iscontinuous(matrix, column)

Return true if the specified column in the given matrix is continuous, false if nominal
"""
iscontinuous(m::Matrix, col::Int) = isempty(m.enum_to_str[col])

"""
    shuffle!(matrix[, buddy])

Shuffle the rows of the matrix in place. If buddy is passed in, buddy will be
shuffled in place using the same random permutation as matrix.
"""
shuffle!(m::Matrix) = permute!(m, randperm(rows(m)))
function shuffle!(m::Matrix, buddy::Matrix)
	perm = randperm(rows(m))
	permute!(m, perm)
	permute!(buddy, perm)
end

"""
    copymatrix(matrix, rows, columns)

Get the specified portion of `matrix` and return it as a new `Matrix`.

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
	data = map(i->m.rows[i][columns], rows)
	attr_name = m.attr_name[columns]
	str_to_enum = m.str_to_enum[columns]
	enum_to_str = m.enum_to_str[columns]
	Matrix(data, attr_name, str_to_enum, enum_to_str, m.datasetname)
end
copymatrix(m::Matrix, rows::Integer, columns::Integer) = copymatrix(m, rows:rows, columns:columns)
copymatrix(m::Matrix, rows, columns::Integer) = copymatrix(m, rows, columns:columns)
copymatrix(m::Matrix, rows::Integer, columns) = copymatrix(m, rows:rows, columns)

"""
    getrows(matrix, rows)

Get the specified rows from `matrix`, as a `Matrix`. `rows` can be a range or an array of indices, or
[any other index](https://docs.julialang.org/en/stable/manual/arrays/#man-supported-index-types-1)
supported by standard julia indexing.
"""
getrows(m::Matrix, rows) = copymatrix(m, rows, 1:columns(m))

"""
	Split

Holds the result of splitting a set of features and labels into a training set and a test set.

Fields are `trainfeatures`, `trainlabels`, `validationfeatures`, `validationlabels`.
"""
struct Split
	trainfeatures::Matrix
	trainlabels::Matrix
	validationfeatures::Matrix
	validationlabels::Matrix
end

"""
    splitmatrix(features, labels, percenttest)

Split the given matrices into a training set and a validation set.

`percenttest`% of the rows will be put in the validation set and `1-percenttest`%
of the rows are put into the training set. Returns a [`Split`](@ref) object.
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

"""
    Matrix(rows, columns)

Create a matrix with the given number of rows and columns.
"""
function Matrix(rows::Integer, columns::Integer)
	data = map(_ -> zeros(columns), 1:rows)
	attr_name = fill("", columns)
	# kinda hacky, but whatever - creates a new (empty) dictionary for each column
	str_to_enum = map(_ -> Dict{AbstractString,Integer}(), 1:columns)
	enum_to_str = map(_ -> Dict{Integer,AbstractString}(), 1:columns)
	Matrix(data, attr_name, str_to_enum, enum_to_str, "")
end

const numbertypes = Set(["REAL", "CONTINUOUS", "INTEGER", "NUMERIC"])

"""
    loadarff(filename)

Read an arff file and return a `Matrix`
"""
function loadarff(filename::AbstractString)::Matrix
	io = open(filename)
	# skip empty lines and comments at the beginning
	skipchars(isspace, io; linecomment='%')
	# initialize variables
	attr_name = Vector{AbstractString}()
	str_to_enum = Vector{Dict{AbstractString,Integer}}()
	enum_to_str = Vector{Dict{Integer,AbstractString}}()
	datasetname = ""
	# read attributes - break when you get to the data
	while true
		line = readline(io)
		(isempty(line) || line[1] == '%') && continue
		upper = uppercase(line)
		if startswith(upper, "@RELATION")
			# Everything after Relation is the datasetname
			datasetname = split(line; limit=2)[2]
		elseif startswith(upper, "@ATTRIBUTE")
			# Attribute should have three distinct parts - @Attribute, name, and type
			# e.g. "@attribute	'handicapped-infants'	{ 'n', 'y'}"
			# or "@ATTRIBUTE	sepallength	Continuous"
			attribute = split(line, r"\s+"; limit=3, keepempty=false)
			ste = Dict{AbstractString,Integer}()
			ets = Dict{Integer,AbstractString}()
			push!(attr_name, attribute[2])
			push!(str_to_enum, ste)
			push!(enum_to_str, ets)
			# If it's one of the number types, there's no need to do anything with it
			if uppercase(attribute[3]) ∉ numbertypes
				stripped = strip(attribute[3], ['{', '}', ' '])
				values = split(stripped, [' ', ',']; keepempty=false)
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
	mappers = map(dict -> getfloatvalue(dict) ∘ strip, str_to_enum)
	while !eof(io)
		line = readline(io)
		(isempty(line) || line[1] == '%') && continue
		row = map((f,x) -> f(x), mappers, split(line, ','))
		push!(data, row)
	end
	close(io)
	Matrix(data, attr_name, str_to_enum, enum_to_str, datasetname)
end

function getfloatvalue(dict::Dict{AbstractString,Integer})
	function mapvalue(value::AbstractString)
		value = strip(value)
		# the ordering of these if statements is very intentional
		# if "?" is a key in the dictionary, then use the value it maps to there
		# if it's not, it's set to missing whether it's a nominal or a continuous feature
		if haskey(dict, value)
			dict[value]
		elseif value == "?"
			missing
		elseif isempty(dict)
			parse(Float64, value)
		else
			error("Error parsing value $value with dict $dict")
		end
	end
	mapvalue
end

"""
    normalize(matrix[, extrema])

Normalizes `matrix` so that all columns have values between 0 and 1.

If `extrema` is included, the matrix is normalized as though the maximum and
minimum value of each column were the values included in extrema.
This allows for two matrices to be normalized using the same ranges.

This method returns a list of extrema that can then be used to normalize another matrix.
(This is typically only useful when you didn't pass the list of extrema in)
"""
normalize(m::Matrix) = normalize(m, vec(extrema(m, 1)))
function normalize(m::Matrix, extrema::Vector{Tuple{Float64,Float64}})
	colinds = map(c -> valuecount(m, c) == 0, 1:columns(m)) .== 0
	for row in m.rows
		map!(row, row, extrema, colinds) do val, ext, real
			real ? normalizevalue(val, ext) : val
		end
	end
	extrema
end

function normalizevalue(value, extrema)
	min, max = extrema
	(value-min)/(max-min)
end

function Base.show(io::IO, m::Matrix)
	println(io, "@RELATION ", m.datasetname)
	for (name, enum_to_str) in zip(m.attr_name, m.enum_to_str)
		println(io, "@ATTRIBUTE ", name, " ", begin
			if isempty(enum_to_str)
				"CONTINUOUS"
			else
				"{", join(values(enum_to_str), ", "), "}"
			end
		end...)
	end
	println(io, "@DATA")
	for row in m.rows
		mapped = map(row, m.enum_to_str) do val, map
			ismissing(val) ? "?" : isempty(map) ? val : map[val]
		end
		join(io, mapped, ", ")
		println(io)
	end
end
