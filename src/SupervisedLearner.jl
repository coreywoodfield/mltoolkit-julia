
# exports for MLToolkit
export SupervisedLearner, train, predict, meansquarederror, measureaccuracy

abstract type SupervisedLearner end

"""
    measureaccuracy(learner, features, labels[, confusion])

Measure the accuracy of a learner given a set of features and labels.

If the problem is a classification problem, this method returns the percent
of instances which are classified correctly. If it is a regression problem,
it calculates the root mean squared error.

If a confusion matrix is included, and the problem is a classification problem,
the confusion matrix will be updated to show common misclassifications.
"""
function measureaccuracy(learner::SupervisedLearner, features::Matrix, labels::Matrix, confusion::Nullable{Matrix}=Nullable{Matrix}())
	if rows(features) != rows(labels)
		error("Expected the features and labels to have the same number of rows")
	elseif columns(labels) != 1
		error("Sorry, this method currently only supports one-dimensional labels")
	elseif rows(features) == 0
		error("Expected at least one row")
	end

	if iscontinuous(labels, 1)
		# The label is continuous, so measure root mean squared error
		sse = 0.
		foreach(features, labels) do feature, target
			prediction = predict(learner, feature)
			delta = target[1] - prediction
			sse += delta^2
		end
		√(sse / rows(features))
	else
		correctcount = 0
		labelvalues = valuecount(labels, 1)
		foreach(features, labels[:,1]) do feature, target
			if target >= labelvalues
				error("The label is out of range")
			end
			prediction = Int(predict(learner, feature))
			if !isnull(confusion)
				get(confusion)[Int(target) + 1, prediction + 1] += 1
			end
			correctcount += prediction == target
		end
		correctcount / rows(features)
	end
end

"""
    meansquarederror(learner, features, labels)

Calculate the mean squared error of a learner given a set of features and a
corresponding set of labels.
"""
function meansquarederror end

"""
    train(learner, features, labels)

Train a learner given a matrix of features and the corresponding labels
"""
function train end

"""
    predict(learner, instance)

Use a learner to predict the class of a specific instance.
"""
function predict end
