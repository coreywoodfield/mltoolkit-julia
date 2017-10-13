#!/usr/bin/env julia

module MLToolkit

# declare Matrix class and constructors
include("Matrix.jl")
# declare SupervisedLearner abstract type
include("SupervisedLearner.jl")
# get arg parsing stuff
include("ParseArgs.jl")
# BaselineLearner class
include("BaselineLearner.jl")

function getlearner(model::AbstractString)::SupervisedLearner
	if model == "baseline"
		BaselineLearner()
	# elseif model == "perceptron"
	# 	Perceptron()
	# elseif model == "neuralnet"
	# 	NeuralNet()
	# elseif model == "decisiontree"
	# 	DecisionTree()
	# elseif model == "knn"
	# 	InstanceBasedLearner()
	else
		error("Unrecognized model: $model")
	end
end

function run()
	srand() # No seed for non-deterministic results (this seeds the global random number generator)
	# srand(1234) # Use a seed for deterministic results (makes debugging easier)
	args = ParseArgs(ARGS)
	learner = getlearner(args.learner)
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

	trainandtest(learner, matrix, evalmode, args.verbose)
	println()
end

function trainandtest(learner::SupervisedLearner, data::Matrix, ::Training, verbose::Bool)
	println("Calculating accuracy on training set...")
	features = copymatrix(data, 1, 1, rows(data), columns(data) - 1)
	labels = copymatrix(data, 1, columns(data) - 1, rows(data), 1)
	labelvalues = valuecount(labels, 1)
	confusion = Nullable(Matrix(labelvalues, labelvalues))
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
	features = copymatrix(data, 1, 1, rows(data), columns(data) - 1)
	labels = copymatrix(data, 1, columns(data) - 1, rows(data), 1)
	elapsedtime = @elapsed train(learner, features, labels)
	println("Time to train (in seconds): ", elapsedtime)
	trainaccuracy = measureaccuracy(features, labels, null)
	println("Training set accuracy: ", trainaccuracy)
	testfeatures = copymatrix(testdata, 1, 1, rows(testdata), columns(testdata) - 1)
	testlabels = copymatrix(testdata, 1, columns(testdata) - 1, rows(testdata), 1)
	labelvalues = valuecount(labels, 1)
	confusion = Nullable(Matrix(labelvalues, labelvalues))
	testAccuracy = measureaccuracy(learner, testfeatures, testlabels, confusion)
	println("Test set accuracy: ", testAccuracy)
	if verbose
		println("\nConfusion matrix: (Row=target value, Col=predicted value)")
		show(get(confusion))
		println("\n")
	end
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Random, verbose::Bool)
	println("Calculating accuracy on a random hold-out set...")
	trainpercent = evalmode.percenttest
	if trainPercent < 0 || trainPercent > 1
		error("Percentage for random evaluation must be between 0 and 1")
	end
	println("Percentage used for training: ", trainpercent)
	println("Percentage used for testing: ", 1 - trainpercent)
	shuffle!(data)
	trainsize = trunc(Int, trainPercent * rows(data))
	trainfeatures = copymatrix(data, 1, 1, trainsize, columns(data) - 1)
	trainlabels = copymatrix(data, 1, columns(data) - 1, trainSize, 1)
	testfeatures = copymatrix(data, trainSize + 1, 1, rows(data) - trainSize, columns(data) - 1)
	testlabels = copymatrix(data, trainSize + 1, columns(data) - 1, rows(data) - trainSize, 1)
	elapsedtime = @elapsed train(learner, trainfeatures, trainlabels)
	println("Time to train (in seconds): ", elapsedtime)
	trainaccuracy = measureaccuracy(learner, trainfeatures, trainlabels)
	println("Training set accuracy: ", trainAccuracy)
	labelvalues = valuecount(labels, 1)
	confusion = Nullable(Matrix(labelvalues, labelvalues))
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
	if folds <= 0
		error("Number of folds must be greater than 0")
	end
	println("Number of folds: ", folds)
	reps = 1
	sumaccuracy = 0.0
	elapsedtime = 0.0
	for j in 1:reps
		shuffle!(data)
		for i in 1:folds
			begin_i = i * rows(data) / folds
			end_i = (i + 1) * rows(data) / folds
			trainFeatures = copymatrix(data, 1, 1, begin_i, columns(data) - 1)
			trainLabels = copymatrix(data, 1, columns(data) - 1, begin_1, 1)
			testFeatures = copymatrix(data, begin_i, 1, end_i - begin_i, columns(data) - 1)
			testLabels = copymatrix(data, begin_i, columns(data) - 1, end_i - begin_i, 1)
			trainFeatures.add(data, end_i, 0, rows(data) - end_i)
			trainLabels.add(data, end_i, columns(data) - 1, data.rows() - end_i)
			elapsedtime += @elapsed learner.train(trainFeatures, trainLabels)
			accuracy = measureAccuracy(learner, testFeatures, testLabels)
			sumAccuracy += accuracy
			println("Rep=", j, ", Fold=", i, ", Accuracy=", accuracy)
		end
	end
	println("Average time to train (in seconds): ", elapsedime / (reps * folds))
	println("Mean accuracy=", sumAccuracy / (reps * folds))
end

end # module

# only run anything if it's being run from the command line
if !isinteractive()
	MLToolkit.run()
end
