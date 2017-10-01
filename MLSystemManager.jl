module MLToolkit

abstract type SupervisedLearner end

include("ParseArgs.jl")
include("Matrix.jl")
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

end



end
