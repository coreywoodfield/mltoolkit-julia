
abstract type SupervisedLearner end

function measureaccuracy(learner::SupervisedLearner, features::Matrix, labels::Matrix, confusion::Matrix)
	if rows(features) != rows(labels)
		error("Expected the features and labels to have the same number of rows")
	elseif columns(labels) != 1
		error("Sorry, this method currently only supports one-dimensional labels")
	elseif rows(features) == 0
		error("Expected at least one row")
	end

	labelvalues = valuecount(labels, 1)
	if labelvalues == 0
		# The label is continuous, so measure root mean squared error
		prediction = Vector{Float64}(1)
		sse = 0.
		for (feature, target) in zip(features, labels)
			prediction[1] = 0. # make sure the prediction is not biased by a previous prediction
			predict(learner, feature, prediction)
			delta = target[1] - prediction[1]
			sse += delta^2
		end
		√(sse / rows(features))
	else
		prediction = Vector{Float64}(1)
		correctcount = 0
		for (feature, target) in zip(features, labels)
			if target[1] >= labelvalues
				error("The label is out of range")
			end
			predict(learner, feature, prediction)
			# TODO Add confusion stuff
			# if(confusion != null)
			# 	confusion.set(targ, pred, confusion.get(targ, pred) + 1);
			if prediction[1] == target[1]
				correctcount += 1
			end
		end
		correctcount / rows(features)
	end
end
