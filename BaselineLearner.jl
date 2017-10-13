"""
For nominal labels, this model simply returns the majority class. For
continuous labels, it returns the mean value.
If the learning model you're using doesn't do as well as this one,
it's time to find a new learning model.
"""
mutable struct BaselineLearner <: SupervisedLearner
	label::Float64
	BaselineLearner() = new()
end

function train(learner::BaselineLearner, features::Matrix, labels::Matrix)
	# labels should just have 1 column, as far as I can tell
	learner.label = valuecount(labels, 1) == 0 ? columnmean(labels, 1) : mostcommonvalue(labels, 1)
end

function predict(learner::BaselineLearner, features::Vector{Float64})
	learner.label
end
