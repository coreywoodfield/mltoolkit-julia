module BaselineModule

# import all the exported things from MLToolkit, so we can extend SupervisedLearner
# and use the MLToolkit.Matrix and related functions without needing to qualify them
importall MLToolkit

# export BaselineLearner so MLToolkit can add `using .BaselineModule` and
# use the learner without qualifying it
export BaselineLearner

"""
    BaselineLearner

For nominal labels, this model simply returns the majority class. For
continuous labels, it returns the mean value.
If the learning model you're using doesn't do as well as this one,
it's time to find a new learning model.
"""
mutable struct BaselineLearner <: SupervisedLearner
	label::Float64
	# sample can be passed in from the command line as follows:
	# ./MLSystemManager -L baseline -A data/iris.arff -E training --sample nondefault
	# if it's not passed in from the command line it will be default
	BaselineLearner(;sample="default") = new()
end

function train(learner::BaselineLearner, ::Matrix, labels::Matrix)
	learner.label = iscontinuous(labels, 1) ? columnmean(labels, 1) : mostcommonvalue(labels, 1)
end

predict(learner::BaselineLearner, ::Row) = learner.label

end
