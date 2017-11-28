abstract type EvalMode end

struct Training <: EvalMode end

struct Random <: EvalMode
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
Base.show(io::IO, ::Random) = print(io, "random")
Base.show(io::IO, ::Cross) = print(io, "cross")
Base.show(io::IO, ::Static) = print(io, "static")

function EvalMode(args::Array)
	mode = shift!(args)
	if mode == "training"
		Training()
	elseif mode == "random"
		Random(parse(Float64, shift!(args)))
	elseif mode == "cross"
		Cross(parse(Int, shift!(args)))
	elseif mode == "static"
		Static(shift!(args))
	end
end

struct ParseArgs
	verbose::Bool
	normalize::Bool
	arff::AbstractString
	learner::AbstractString
	evaluation::EvalMode
	other::Dict{Symbol,AbstractString}
end

function parseargs(args::Array)
	verbose, normalize, arff, learner, evaluation, other = false, false, "", "", nothing, Dict()
	try
		while !isempty(args)
			option = shift!(args)
			if option == "-V"
				verbose = true
			elseif option == "-N"
				normalize = true
			elseif option == "-A"
				arff = shift!(args)
			elseif option == "-L"
				learner = shift!(args)
			elseif option == "-E"
				evaluation = EvalMode(args)
			elseif option == "--seed"
				srand(parse(shift!(args)))
			elseif startswith(option, "--")
				other[Symbol(option[3:end])] = shift!(args)
			else
				error("Invalid parameter: $option")
			end
		end
		if arff != "" && learner != "" && evaluation != nothing
			return ParseArgs(verbose, normalize, arff, learner, evaluation, other)
		end
	end
	println("Usage:")
	println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n")
	println("OPTIONS:")
	println("-V Print the confusion matrix and learner accuracy on individual class values")
	println("-N Use normalized data")
	println()
	println("Possible evaluation methods are:")
	println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training")
	println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]")
	println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]")
	println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n")
	exit(1)
end
