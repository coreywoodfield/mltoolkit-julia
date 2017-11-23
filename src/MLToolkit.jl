module MLToolkit

export Σ

const Σ = sum

# declare Matrix class and constructors
include("Matrix.jl")
# declare SupervisedLearner abstract type
include("SupervisedLearner.jl")
# get arg parsing stuff
include("ParseArgs.jl")
# BaselineLearner class
include("BaselineLearner.jl")

# import BaselineLearner from BaselineModule
using .BaselineModule

learners = Dict("baseline" => BaselineLearner)

function getlearner(model::AbstractString, args::Dict{Symbol,AbstractString})::SupervisedLearner
	if haskey(learners, model)
		# pass extra command line args to the learner constructor as keyword arguments
		learners[model](;args...)
	else
		error("Unrecognized model: $model. Supported models are $(join(keys(learners), ", ", ", and "))")
	end
end

function run()
	srand() # No seed for non-deterministic results (this seeds the global random number generator)
	# srand(1234) # Use a seed for deterministic results (makes debugging easier)
	args = parseargs(ARGS)
	learner = getlearner(args.learner, args.other)
	evalmode = args.evaluation

	matrix = loadarff(args.arff)
	if args.normalize
		println("Using normalized data\n")
		extrema = normalize(matrix)
	end
	# load (and normalize) test data as well if the evalmode is Static
	if evalmode isa Static
		evalmode.data = loadarff(evalmode.filename)
		if args.normalize
			normalize(evalmode.data, extrema)
		end
	end

	println()
	println("Dataset name: ", matrix.datasetname)
	println("Number of instances: ", rows(matrix))
	println("Number of attributes: ", columns(matrix))
	println("Learning algorithm: ", args.learner)
	println("Evaluation method: ", evalmode)
	println()

	# precompile so that the compilation time doesn't skew the first runtime, if we're timing anything
	precompile(train, (typeof(learner), Matrix, Matrix))
	precompile(measureaccuracy, (typeof(learner), Matrix, Matrix))
	precompile(measureaccuracy, (typeof(learner), Matrix, Matrix, Nullable{Matrix}))
	trainandtest(learner, matrix, evalmode, args.verbose)
	println()
end

function trainandtest(learner::SupervisedLearner, data::Matrix, ::Training, verbose::Bool)
	println("Calculating accuracy on training set...")
	features = copymatrix(data, 1:rows(data), 1:columns(data)-1)
	labels = copymatrix(data, 1:rows(data), columns(data))
	confusion = initconfusionmatrix(labels)
	elapsedtime = @elapsed train(learner, features, labels)
	println("Time to train (in seconds): ", elapsedtime)
	accuracy = measureaccuracy(learner, features, labels, confusion)
	println("Training set accuracy: ", accuracy)
	if verbose
		println("\nConfusion matrix: (Row=target value, Col=predicted value)")
		show(get(confusion))
		println("\n")
	end
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Static, verbose::Bool)
	testdata = evalmode.data
	println("Calculating accuracy on separate test set...")
	println("Test set name: ", evalmode.filename)
	println("Number of test instances: ", rows(testdata))
	features = copymatrix(data, 1:rows(data), 1:columns(data)-1)
	labels = copymatrix(data, 1:rows(data), columns(data))
	elapsedtime = @elapsed train(learner, features, labels)
	println("Time to train (in seconds): ", elapsedtime)
	trainaccuracy = measureaccuracy(learner, features, labels)
	println("Training set accuracy: ", trainaccuracy)
	testfeatures = copymatrix(testdata, 1:rows(testdata), 1:columns(testdata)-1)
	testlabels = copymatrix(testdata, 1:rows(testdata), columns(testdata))
	confusion = initconfusionmatrix(labels)
	testaccuracy = measureaccuracy(learner, testfeatures, testlabels, confusion)
	println("Test set accuracy: ", testaccuracy)
	if verbose
		println("\nConfusion matrix: (Row=target value, Col=predicted value)")
		show(get(confusion))
		println("\n")
	end
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Random, verbose::Bool)
	println("Calculating accuracy on a random hold-out set...")
	trainpercent = evalmode.percenttest
	if trainpercent < 0 || trainpercent > 1
		error("Percentage for random evaluation must be between 0 and 1")
	end
	println("Percentage used for training: ", trainpercent)
	println("Percentage used for testing: ", 1 - trainpercent)
	shuffle!(data)
	trainsize = trunc(Int, trainpercent * rows(data))
	trainfeatures = copymatrix(data, 1:trainsize, 1:columns(data)-1)
	trainlabels = copymatrix(data, 1:trainsize, columns(data))
	testfeatures = copymatrix(data, trainsize+1:rows(data), 1:columns(data)-1)
	testlabels = copymatrix(data, trainsize+1:rows(data), columns(data))
	elapsedtime = @elapsed train(learner, trainfeatures, trainlabels)
	println("Time to train (in seconds): ", elapsedtime)
	trainaccuracy = measureaccuracy(learner, trainfeatures, trainlabels)
	println("Training set accuracy: ", trainaccuracy)
	confusion = initconfusionmatrix(testlabels)
	testaccuracy = measureaccuracy(learner, testfeatures, testlabels, confusion)
	println("Test set accuracy: ", testaccuracy)
	if verbose
		println("\nConfusion matrix: (Row=target value, Col=predicted value)")
		show(get(confusion))
		println("\n")
	end
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Cross, verbose::Bool)
	println("Calculating accuracy using cross-validation...")
	folds = evalmode.numfolds
	folds <= 0 && error("Number of folds must be greater than 0")
	println("Number of folds: ", folds)
	reps = 1
	sumaccuracy = 0.0
	elapsedtime = 0.0
	for j in 1:reps
		shuffle!(data)
		for i in 0:folds-1
			begin_i = i * div(rows(data), folds)
			end_i = (i + 1) * div(rows(data), folds)
			trainfeatures = copymatrix(data, [1:begin_i; end_i+1:rows(data)], 1:columns(data)-1)
			trainlabels = copymatrix(data, [1:begin_i; end_i+1:rows(data)], columns(data))
			testfeatures = copymatrix(data, begin_i+1:end_i, 1:columns(data)-1)
			testlabels = copymatrix(data, begin_i+1:end_i, columns(data))
			elapsedtime += @elapsed train(learner, trainfeatures, trainlabels)
			accuracy = measureaccuracy(learner, testfeatures, testlabels)
			sumaccuracy += accuracy
			println("Rep=", j, ", Fold=", i, ", Accuracy=", accuracy)
		end
	end
	println("Average time to train (in seconds): ", elapsedtime / (reps * folds))
	println("Mean accuracy=", sumaccuracy / (reps * folds))
end

function initconfusionmatrix(labelmatrix::Matrix)
	dims = valuecount(labelmatrix, 1)
	confusion = Matrix(dims, dims)
	for i in 1:dims
		setattributename(confusion, i, attributevalue(labelmatrix, 1, i-1))
	end
	Nullable(confusion)
end

end # module
