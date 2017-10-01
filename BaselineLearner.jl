"""
For nominal labels, this model simply returns the majority class. For
continuous labels, it returns the mean value.
If the learning model you're using doesn't do as well as this one,
it's time to find a new learning model.
"""
struct BaselineLearner <: SupervisedLearner
	labels::Vector{Float64}
	BaselineLearner() = new([])
end

function train(learner::BaselineLearner, features::Matrix, labels::Matrix)
	for i in 1:columns(labels)
		push!(learner.labels, valuecount(labels, i) == 0
								? columnmean(labels, i)
								: mostcommonvalue(labels, i))
	end
end

function predict(learner::BaselineLearner, features::Vector{Float64}, labels::Vector{Float64})
	copy!(labels, learner.labels)
end
