module MLToolkit

using LinearAlgebra, Random

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
	# If you want to use a specific seed for the RNG, use `--seed <seed>` on the command line
	Random.seed!() # this seeds the global RNG. If you pass in seed it gets reseeded in parseargs
	args = parseargs(ARGS)
	learner = getlearner(args.learner, args.other)
	evalmode = args.evaluation

	matrix = loadarff(args.arff)
	if args.normalize
		@info "Using normalized data"
		extrema = normalize(matrix)
	end
	# load (and normalize) test data as well if the evalmode is Static
	if evalmode isa Static
		evalmode.data = loadarff(evalmode.filename)
		if args.normalize
			normalize(evalmode.data, extrema)
		end
	end

	@info "Dataset" matrix.datasetname rows(matrix) columns(matrix) args.learner evalmode

	# precompile so that the compilation time doesn't skew the first runtime, if we're timing anything
	precompile(train, (typeof(learner), Matrix, Matrix))
	precompile(measureaccuracy, (typeof(learner), Matrix, Matrix))
	precompile(measureaccuracy, (typeof(learner), Matrix, Matrix, Union{Matrix,Nothing}))
	trainandtest(learner, matrix, evalmode)
end

function trainandtest(learner::SupervisedLearner, data::Matrix, ::Training)
	@info "Calculating accuracy on training set"
	features = copymatrix(data, 1:rows(data), 1:columns(data)-1)
	labels = copymatrix(data, 1:rows(data), columns(data))
	confusion = initconfusionmatrix(labels)
	elapsedtime = @elapsed train(learner, features, labels)
	accuracy = measureaccuracy(learner, features, labels, confusion)
	@info "Results" timetotrain=elapsedtime accuracy
	@debug "Confusion matrix: (Row=target value, Col=predicted value)" confusion
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Static)
	testdata = evalmode.data
	@info "Calculating accuracy on separate test set" evalmode.filename rows(testdata)
	features = copymatrix(data, 1:rows(data), 1:columns(data)-1)
	labels = copymatrix(data, 1:rows(data), columns(data))
	elapsedtime = @elapsed train(learner, features, labels)
	trainaccuracy = measureaccuracy(learner, features, labels)
	testfeatures = copymatrix(testdata, 1:rows(testdata), 1:columns(testdata)-1)
	testlabels = copymatrix(testdata, 1:rows(testdata), columns(testdata))
	confusion = initconfusionmatrix(labels)
	testaccuracy = measureaccuracy(learner, testfeatures, testlabels, confusion)
	@info "Results" timetotrain=elapsedtime trainaccuracy testaccuracy
	@debug "Confusion matrix: (Row=target value, Col=predicted value)" confusion
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Rando)
	trainpercent = evalmode.percenttest
	if trainpercent < 0 || trainpercent > 1
		error("Percentage for random evaluation must be between 0 and 1")
	end
	@info "Calculating accuracy on a random hold-out set" trainpercent testpercent=1-trainpercent
	shuffle!(data)
	trainsize = trunc(Int, trainpercent * rows(data))
	trainfeatures = copymatrix(data, 1:trainsize, 1:columns(data)-1)
	trainlabels = copymatrix(data, 1:trainsize, columns(data))
	testfeatures = copymatrix(data, trainsize+1:rows(data), 1:columns(data)-1)
	testlabels = copymatrix(data, trainsize+1:rows(data), columns(data))
	elapsedtime = @elapsed train(learner, trainfeatures, trainlabels)
	trainaccuracy = measureaccuracy(learner, trainfeatures, trainlabels)
	confusion = initconfusionmatrix(testlabels)
	testaccuracy = measureaccuracy(learner, testfeatures, testlabels, confusion)
	@info "Results" timetotrain=elapsedtime trainaccuracy testaccuracy
	@debug "Confusion matrix: (Row=target value, Col=predicted value)" confusion
end

function trainandtest(learner::SupervisedLearner, data::Matrix, evalmode::Cross)
	folds = evalmode.numfolds
	folds <= 0 && error("Number of folds must be greater than 0")
	@info "Calculating accuracy using cross-validation" folds
	sumaccuracy = 0.0
	elapsedtime = 0.0
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
		@info "Fold $(i+1)" testset=begin_i+1:end_i accuracy
	end
	@info "Average results" timetotrain=elapsedtime/folds accuracy=sumaccuracy/folds
end

function initconfusionmatrix(labelmatrix::Matrix)
	dims = valuecount(labelmatrix, 1)
	confusion = Matrix(dims, dims)
	for i in 1:dims
		setattributename(confusion, i, attributevalue(labelmatrix, 1, i-1))
	end
	confusion
end

end # module
