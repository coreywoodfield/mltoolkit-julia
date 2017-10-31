
abstract type SupervisedLearner end

function measureaccuracy(learner::SupervisedLearner, features::Matrix, labels::Matrix, confusion::Nullable{Matrix}=Nullable{Matrix}())
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
		sse = 0.
		for (feature, target) in zip(features, labels)
			prediction = predict(learner, feature)
			delta = target[1] - prediction
			sse += delta^2
		end
		√(sse / rows(features))
	else
		correctcount = 0
		for (feature, target) in zip(features, labels)
			if target[1] >= labelvalues
				error("The label is out of range")
			end
			prediction = Int(predict(learner, feature))
			if !isnull(confusion)
				get(confusion)[Int(target[1]) + 1, prediction + 1] += 1
			end
			if prediction == target[1]
				correctcount += 1
			end
		end
		correctcount / rows(features)
	end
end

function meansquarederror(learner, features::Matrix, labels::Matrix)
	error("meansquarederror not implemented for $(typeof(learner))")
end

function train(learner, features::Matrix, labels::Matrix)
	error("train not implemented for $(typeof(learner))")
end

function predict(learner, features::Vector{Float64})
	error("predict not implemented for $(typeof(learner))")
end
