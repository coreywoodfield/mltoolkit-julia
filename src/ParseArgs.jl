abstract type EvalMode end

struct Training <: EvalMode end

struct Rando <: EvalMode
	percenttest::AbstractFloat
end

struct Cross <: EvalMode
	numfolds::Integer
end

mutable struct Static <: EvalMode
	filename::AbstractString
	data::Matrix
	Static(filename) = new(filename)
end

Base.show(io::IO, ::Training) = print(io, "training")
Base.show(io::IO, ::Rando) = print(io, "random")
Base.show(io::IO, ::Cross) = print(io, "cross")
Base.show(io::IO, ::Static) = print(io, "static")

function EvalMode(args::Array)
	mode = popfirst!(args)
	if mode == "training"
		Training()
	elseif mode == "random"
		Rando(parse(Float64, popfirst!(args)))
	elseif mode == "cross"
		Cross(parse(Int, popfirst!(args)))
	elseif mode == "static"
		Static(popfirst!(args))
	end
end

struct ParseArgs
	normalize::Bool
	arff::AbstractString
	learner::AbstractString
	evaluation::EvalMode
	other::Dict{Symbol,AbstractString}
end

function parseargs(args::Array)
	normalize = false
	arff = ""
	learner = ""
	evaluation = nothing
	other = Dict()
	try
		while !isempty(args)
			option = popfirst!(args)
			if option == "-V"
				ENV["JULIA_DEBUG"] = "MLToolkit"
			elseif option == "-N"
				normalize = true
			elseif option == "-A"
				arff = popfirst!(args)
			elseif option == "-L"
				learner = popfirst!(args)
			elseif option == "-E"
				evaluation = EvalMode(args)
			elseif option == "--seed"
				Random.seed!(parse(popfirst!(args)))
			elseif startswith(option, "--")
				other[Symbol(option[3:end])] = popfirst!(args)
			else
				error("Invalid parameter: $option")
			end
		end
		if arff != "" && learner != "" && evaluation != nothing
			return ParseArgs(normalize, arff, learner, evaluation, other)
		end
	catch
	end
	@error join(["Usage:",
	"MLSystemManager -L learningAlgorithm -A ARFF_File -E evaluationMethod [extraParameters] [OPTIONS]",
	"OPTIONS:",
	"-V Print the confusion matrix and learner accuracy on individual class values",
	"-N Use normalized data",
	"",
	"Possible evaluation methods are:",
	"-E training",
	"-E static testARFF_File",
	"-E random training_percentage",
	"-E cross numOfFolds"], "\n")
	exit(1)
end
