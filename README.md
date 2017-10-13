# mltoolkit-julia
Julia port of the BYU CS 478 machine learning toolkit

## How to extend this toolkit
If you want to add a learner, create a new file in the project directory for the learner.
Create the learner type and have it extend the `SupervisedLearner` type (e.g. `struct Perceptron <: SupervisedLearner ...`).
Implement the `train(learner::YourLearner, features::Matrix, labels::Matrix)` function and the
`predict(learner::YourLearner, features::Vector{Float64})` function for your learner. The `predict` function,
instead of modifying an array passed in (as the java toolkit does), simply returns the prediction.
Add `include("YourFile.jl")` in MLSystemManager.jl under all the rest of the includes at the top of the file,
and add a case to the `getlearner` function for your learner.

## How to run this toolkit
When you run the MLSystemManager.jl file from the command line (non-interactively), it will run just as the java toolkit does.
It takes the same options as the java toolkit. If I were in the root directory of the project, and wanted to run the baseline
learner on a arff file in that directory in the training evaluation mode, I would do so with the following command:
`./MLSystemManager.jl -L baseline -A iris.arff -E training`

## Current status of the toolkit
As far as I can tell, the toolkit works. There's a good chance there are bugs; I will fix them as I come across them.
I also intend to gradually edit the code so that it feels more like a native Julia program, and less like a Julia program
that I ported from Java.

## Important things to note
Julia indexing starts at 1 instead of 0. The `Matrix` class I created is 1-indexed, and most of the functions that involve
the `Matrix` class also use 1-indexing.
