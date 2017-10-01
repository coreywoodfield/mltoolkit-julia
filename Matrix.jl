
const numbertypes = Set(["REAL", "CONTINUOUS", "INTEGER"])
const MISSING = Inf

"""
    Matrix <: AbstractMatrix{Float64}

2-d matrix containing arff data.

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
end

# These functions allow the Matrix to act like a standard julia collection
# iterate over rows using `for row in m ... end`
# get a specific row using `m[1]` or a specific value using `m[1,5]`
Base.start(m::Matrix) = start(m.data)
Base.next(m::Matrix, state) = next(m.data, state)
Base.done(m::Matrix, state) = done(m.data, state)
Base.eltype(::Type{Matrix}) = Vector{Float64}
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
shuffle!(m::Matrix, rng::AbstractRNG) = permute!(m, randperm(rng, rows(m)))
function shuffle!(m::Matrix, rng::AbstractRNG, buddy::Matrix)
	perm = randperm(rng, rows(m))
	permute!(m, perm)
	permute!(buddy, perm)
end

function loadarff(filename::AbstractString)::Matrix
	io = open(filename)
	# skip empty lines and comments at the beginning
	skipchars(io, isspace; linecomment='%')
	# initialize variables
	attr_name = Vector{AbstractString}()
	str_to_enum = Vector{Dict{AbstractString,Integer}}()
	enum_to_str = Vector{Dict{Integer,AbstractString}}()
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
				i = 0
				for val in values
					ste[val] = i
					ets[i] = val
					i += 1
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
	Matrix(data, attr_name, str_to_enum, enum_to_str)
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
