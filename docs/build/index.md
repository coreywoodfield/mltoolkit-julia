
<a id='MLToolkit.jl-Documentation-1'></a>

# MLToolkit.jl Documentation



<a id='MLToolkit.train' href='#MLToolkit.train'>#</a>
**`MLToolkit.train`** &mdash; *Function*.



```
train(learner, features, labels)
```

Train a learner given a matrix of features and the corresponding labels


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/SupervisedLearner.jl#L62-L66' class='documenter-source'>source</a><br>

<a id='MLToolkit.predict' href='#MLToolkit.predict'>#</a>
**`MLToolkit.predict`** &mdash; *Function*.



```
predict(learner, instance)
```

Use a learner to predict the class of a specific instance.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/SupervisedLearner.jl#L69-L73' class='documenter-source'>source</a><br>

<a id='MLToolkit.measureaccuracy' href='#MLToolkit.measureaccuracy'>#</a>
**`MLToolkit.measureaccuracy`** &mdash; *Function*.



```
measureaccuracy(learner, features, labels[, confusion])
```

Measure the accuracy of a learner given a set of features and labels.

If the problem is a classification problem, this method returns the percent of instances which are classified correctly. If it is a regression problem, it calculates the root mean squared error.

If a confusion matrix is included, and the problem is a classification problem, the confusion matrix will be updated to show common misclassifications.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/SupervisedLearner.jl#L7-L18' class='documenter-source'>source</a><br>

<a id='MLToolkit.meansquarederror' href='#MLToolkit.meansquarederror'>#</a>
**`MLToolkit.meansquarederror`** &mdash; *Function*.



```
meansquarederror(learner, features, labels)
```

Calculate the mean squared error of a learner given a set of features and a corresponding set of labels.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/SupervisedLearner.jl#L54-L59' class='documenter-source'>source</a><br>

<a id='MLToolkit.Matrix' href='#MLToolkit.Matrix'>#</a>
**`MLToolkit.Matrix`** &mdash; *Type*.



```
Matrix <: AbstractMatrix{Float64}
```

2-d matrix (typically) containing data from an arff file

To read a matrix in from an ARFF file, use [`loadarff`](index.md#MLToolkit.loadarff). To copy part of an existing matrix, use [`copymatrix`](index.md#MLToolkit.copymatrix). To initialize an empty matrix with a certain size, use [`Matrix(rows, columns)`](index.md#MLToolkit.Matrix)

You can iterate over the rows using a standard for loop:

```julia
for row in matrix
	...
end
```

Rows and individual elements may be accessed or set using index notation:

```julia
row1 = matrix[1]
element5 = matrix[1,5]
matrix[1,5] = 1.0
```


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L10-L32' class='documenter-source'>source</a><br>

<a id='MLToolkit.loadarff' href='#MLToolkit.loadarff'>#</a>
**`MLToolkit.loadarff`** &mdash; *Function*.



```
loadarff(filename)
```

Read an arff file and return a `Matrix`


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L194-L198' class='documenter-source'>source</a><br>

<a id='MLToolkit.copymatrix' href='#MLToolkit.copymatrix'>#</a>
**`MLToolkit.copymatrix`** &mdash; *Function*.



```
copymatrix(matrix, rows, columns)
```

Get the specified portion of `matrix` and return it as a new `Matrix`.

`rows` and `columns` should be a range or an array of indices, or [any other index](https://docs.julialang.org/en/stable/manual/arrays/#man-supported-index-types-1) supported by standard julia indexing.

This gets a view of the original matrix, and not a copy. If you modify this matrix, the original matrix will also be modified.

!!! note
    The `Matrix` class is 1-indexed. Thus, the values for rows and columns should be between 1 and the number of rows or columns, inclusively.



<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L108-L124' class='documenter-source'>source</a><br>

<a id='MLToolkit.getrows' href='#MLToolkit.getrows'>#</a>
**`MLToolkit.getrows`** &mdash; *Function*.



```
getrows(matrix, rows)
```

Get the specified rows from `matrix`, as a `Matrix`. `rows` can be a range or an array of indices, or [any other index](https://docs.julialang.org/en/stable/manual/arrays/#man-supported-index-types-1) supported by standard julia indexing.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L136-L142' class='documenter-source'>source</a><br>

<a id='MLToolkit.splitmatrix' href='#MLToolkit.splitmatrix'>#</a>
**`MLToolkit.splitmatrix`** &mdash; *Function*.



```
splitmatrix(features, labels, percenttest)
```

Split the given matrices into a training set and a validation set.

`percenttest`% of the rows will be put in the validation set and `1-percenttest`% of the rows are put into the training set. Returns a [`Split`](index.md#MLToolkit.Split) object.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L159-L166' class='documenter-source'>source</a><br>

<a id='MLToolkit.Split' href='#MLToolkit.Split'>#</a>
**`MLToolkit.Split`** &mdash; *Type*.



```
Split
```

Holds the result of splitting a set of features and labels into a training set and a test set.

Fields are `trainfeatures`, `trainlabels`, `validationfeatures`, `validationlabels`.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L145-L151' class='documenter-source'>source</a><br>

<a id='MLToolkit.shuffle!' href='#MLToolkit.shuffle!'>#</a>
**`MLToolkit.shuffle!`** &mdash; *Function*.



```
shuffle!(matrix[, buddy])
```

Shuffle the rows of the matrix in place. If buddy is passed in, buddy will be shuffled in place using the same random permutation as matrix.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L95-L100' class='documenter-source'>source</a><br>

<a id='MLToolkit.normalize' href='#MLToolkit.normalize'>#</a>
**`MLToolkit.normalize`** &mdash; *Function*.



```
normalize(matrix[, extrema])
```

Normalizes `matrix` so that all columns have values between 0 and 1.

If `extrema` is included, the matrix is normalized as though the maximum and minimum value of each column were the values included in extrema. This allows for two matrices to be normalized using the same ranges.

This method returns a list of extrema that can then be used to normalize another matrix. (This is typically only useful when you didn't pass the list of extrema in)


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/Matrix.jl#L264-L275' class='documenter-source'>source</a><br>

<a id='MLToolkit.BaselineModule.BaselineLearner' href='#MLToolkit.BaselineModule.BaselineLearner'>#</a>
**`MLToolkit.BaselineModule.BaselineLearner`** &mdash; *Type*.



```
BaselineLearner
```

For nominal labels, this model simply returns the majority class. For continuous labels, it returns the mean value. If the learning model you're using doesn't do as well as this one, it's time to find a new learning model.


<a target='_blank' href='https://github.com/coreywoodfield/mltoolkit-julia/blob/ec534aeda3a14c32c84e625165ec976d66eff1b9/src/BaselineLearner.jl#L11-L18' class='documenter-source'>source</a><br>

