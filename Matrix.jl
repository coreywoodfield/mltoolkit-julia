
const numbertypes = Set(["REAL", "CONTINUOUS", "INTEGER"])
const MISSING = Inf

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
	data::Vector{Vector{Float64}}
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
Base.setindex!(m::Matrix, v::Vector{Float64}, i::Int) = m.data[i] = v
Base.setindex!(m::Matrix, v::Float64, i::Vararg{Int, 2}) = m.data[i[1]][i[2]] = v

const rows = Base.length
columns(m::Matrix) = length(m.attr_name)
attributename(m::Matrix, col::Integer) = m.attr_name[col]
setattributename(m::Matrix, col::Integer, name::AbstractString) = m.attr_name[col] = name
attributevalue(m::Matrix, col::Integer, value::Integer) = m.enum_to_str[col][value]
valuecount(m::Matrix, col::Integer) = length(m.enum_to_str[col])
columnmean(m::Matrix, col::Integer) = mean(m, 1)[col]
columnminimum(m::Matrix, col::Integer) = minimum(x, 1)[col]
columnmaximum(m::Matrix, col::Integer) = maximum(x, 1)[col]
function mostcommonvalue(m::Matrix, col::Integer)
	counts = Dict{Float64,Integer}()
	for row in m
		value = row[col]
		if value != MISSING
			counts[value] = get(counts, value, 0) + 1
		end
	end
	mcv = MISSING
	maxcount = 0
	for (val, count) in counts
		if count > maxcount
			maxcount = count
			mcv = val
		end
	end
	mcv
end
shuffle!(m::Matrix) = permute!(m, randperm(rows(m)))
function shuffle!(m::Matrix, buddy::Matrix)
	perm = randperm(rows(m))
	permute!(m, perm)
	permute!(buddy, perm)
end

"""
    copymatrix(m, rowstart, colstart, rowcount, colcount)

Copies the specified portion of Matrix `m` and returns it as a new matrix.
`rowstart` and `colstart` should be 1-indexed values

!!! warning

    This differs from the java/c++ version, where `rowstart` and `colstart` are
    0-indexed. Here, if you want to copy the whole matrix `m`, you would call
    `copymatrix(m, 1, 1, rows(m), columns(m))`, not `copymatrix(m, 0, 0, rows(m), columns(m))`
    as you would call in java/c++
"""
function copymatrix(m::Matrix, rowstart::Integer, colstart::Integer, rowcount::Integer, colcount::Integer)::Matrix
	data = Vector{Vector{Float64}}(rowcount)
	# Julia ranges include both end values, so if they just want one column we should do colstart:colstart
	rowoffset, columnoffset = rowstart - 1, colstart - 1
	columns = colstart:columnoffset+colcount
	for i in 1:rowcount
		data[i] = copy(m.data[rowoffset+i][columns])
	end
	attr_name = m.attr_name[columns]
	str_to_enum = m.str_to_enum[columns]
	enum_to_str = m.enum_to_str[columns]
	Matrix(data, attr_name, str_to_enum, enum_to_str, m.datasetname)
end

"""
    add!(matrix1, matrix2, rowstart, colstart, rowcount)

Adds the specified portion of `matrix2` to the end of `matrix1`. `rowstart` and
`colstart` should be 1-indexed values.

!!! warning

    This differs from the java/c++ version, where `rowstart` and `colstart` are
    0-indexed. Here, if you want to add the whole matrix `m` to `n`, you would call
    `add!(n, m, 1, 1, rows(m))`, not `add!(m, 0, 0, rows(m))` as you would call in java/c++
"""
function add!(this::Matrix, that::Matrix, rowstart::Integer, colstart::Integer, rowcount::Integer)
	columnoffset = colstart - 1
	if columnoffset + columns(this) > columns(that)
		error("out of range")
	end
	if any(i -> valuecount(this, i) != valuecount(that, columnoffset+i), 1:columns(this))
		error("Incompatible relations")
	end
	append!(this.data, map(row -> copy(row[colstart:columnoffset+columns(this)]), that.data[rowstart:rowstart+rowcount]))
	nothing
end

function Matrix(rows::Integer, columns::Integer)
	data = Vector{Vector{Float64}}(rows)
	for i in 1:rows
		data[i] = zeros(columns)
	end
	attr_name = fill("", columns)
	# kinda hacky, but whatever - creates a new (empty) dictionary for each column
	str_to_enum = map(x -> Dict{AbstractString,Integer}(), 1:columns)
	enum_to_str = map(x -> Dict{Integer,AbstractString}(), 1:columns)
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
	data = Vector{Vector{Float64}}()
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
	elseif value == "?"
		MISSING
	elseif haskey(dict, value)
		dict[value]
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
    normalize(m, extrema)

Normalizes the matrix `m` using the maximum and minimum values passed in as `extrema`.
This allows for two matrices to be normalized using the same ranges.
`normalize(m)` on the first matrix returns a list of extrema that can be used here
to normalize the second matrix using the same ranges
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

import Base.show

function show(io::IO, m::Matrix)
	println(io, "@RELATION ", m.datasetname)
	for (name, values) in zip(m.attr_name, m.enum_to_str)
		println(io, "@ATTRIBUTE ", name, " ", begin
			valcount = length(values)
			if valcount == 0
				"CONTINUOUS"
			else
				"{", join(map(i -> values[i], 0:valcount-1), ", "), "}"
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
