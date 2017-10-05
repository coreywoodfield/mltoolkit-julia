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

function getLearner(model::AbstractString, rng::AbstractRNG)::SupervisedLearner
	if model == "baseline"
		BaselineLearner()
	# elseif model == "perceptron"
	# 	Perceptron(rng)
	# elseif model == "neuralnet"
	# 	NeuralNet(rng)
	# elseif model == "decisiontree"
	# 	DecisionTree()
	# elseif model == "knn"
	# 	InstanceBasedLearner()
	else
		error("Unrecognized model: $model")
	end
end

function run()
	rng = srand() # No seed for non-deterministic results
	# rng = srand(1234) # Use a seed for deterministic results (makes debugging easier)
	args = ParseArgs(ARGS)
	learner = getLearner(args.learner, rng)

	matrix = loadarff(args.arff)
	if args.normalize
		println("Using normalized data\n")
		normalize(matrix)
	end

	println()
	println("Dataset name: ", matrix.datasetname)
	println("Number of instances: ", rows(matrix))
	println("Number of attributes: ", columns(matrix))
	println("Learning algorithm: ", args.learner)
	println("Evaluation method: ", args.evaluation)
	println()

	trainandtest(learner, matrix, args.evaluation)
	println()
end

function trainandtest(learner::SupervisedLearner, data::Matrix, ::Training)
	println("Calculating accuracy on training set...");
	features = copymatrix(data, 1, 1, rows(data), columns(data) - 1)
	labels = copymatrix(data, 1, columns(data) - 1, rows(data), 1)
	# Matrix confusion = new Matrix();
	elapsedtime = @elapsed train(learner, features, labels)
	println("Time to train (in seconds): ", elapsedtime)
	accuracy = measureaccuracy(learner, features, labels, data)
	println("Training set accuracy: ", accuracy)

	# TODO this stuff
	# if printConfusionMatrix
	# 	println("\nConfusion matrix: (Row=target value, Col=predicted value)")
	# 	show(confusion.print())
	# 	println("\n")
	# end
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Static)

end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Random)

end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Cross)

end

end # module

# only run anything if it's being run from the command line
if !isinteractive()
	MLToolkit.run()
end
