#Tree design doc
Decision tree classifiers are both popular supervised learning algorithms and also building blocks for other ensemble learning algorithms such as random forests, boosting, etc. This document discuss the design for its implementation in the Spark project.

**The current design will be optimized for the scenario where all the data can be fit into the in-cluster memory.**

##Algorithm
Decision tree classifier is formed by creating recursive binary partitions using the optimal splitting criterion that maximizes the information gain at each step. It handles both ordered (numeric) and unordered (categorial) features.

###Identifying Split Predicates
The split predicates will be calculated by performing a single pass over the data  at the start of the tree model building. The binning of the data can be performed using two techniques:

1. Sorting the ordered features and finding the exact quantile points. Complexity: O(N*logN) * #features
2. Using an [approximate quantile calculation algorithm](http://infolab.stanford.edu/~manku/papers/99sigmod-unknown.pdf) cited by the PLANET paper. 

###Optimal Splitting Criterion
The splitting criterion is calculated using one of two popular criterion:

1. [Gini impurity](http://en.wikipedia.org/wiki/Gini_coefficient)
2. [Entropy](http://en.wikipedia.org/wiki/Information_gain_in_decision_trees)

Each split is stored in a model for future predictions.

###Stopping criterion
There are various criterion that can be used to stop adding more levels to the tree. The first implementation will be kept simple and will use the following criteria : no further information gain can be achieved or the maximum depth has been reached. Once a stopping criteria is met, the current node is a leaf of the tree and updates the model with the distribution of the remaining classes at the node.

###Prediction
To make a prediction, a new sample is run through the decision tree model till it arrives at a leaf node. Upon reaching the leaf node, a prediction is made using the distribution of the underlying samples. (typically, the distribution itself is the output)

##Implementation

###Code
For a consistent API, the training code will be consistent with the existing logistic regressions algorithms for supervised learning.

The train() method will take be of the following format

    def train(input: RDD[(Double, Array[Double])]): DecisionTreeModel = {...}
    def predict(testData: spark.RDD[Array[Double]]) = {...}

All splitting criterion can be evaluated in parallel using the *map* operation. The *reduce* operation will select the best splitting criterion. The split criterion will create a *filter* that should be applied to the RDD at each node to derive the RDDs at the next node.

The pseudocode is given below:

    def train(input: RDD[(Double, Array[Double])]): DecisionTreeModel = {
		filterList = new List()
		root = new Node()
		buildTree(root,input,filterList)
	}
	
	def buildTree(node : Node, input : RDD[(Double, Array[Double])]), filterList : List) : Tree = {
		splits = find_possible_splits(input)
		bestSplit = splits.map( split => calculateInformationGain(input, split)).reduce(_ max _)
		if (bestSplit > threshold){
			leftRDD = RDD.filter(sample => sample.satisfiesSplit(bestSplit))
			rightRDD = RDD.filter(sample => sample.satisfiesSplit(bestSplit.invert()))
			node.split = bestSplit
			node.left = new Node()
			node.right = new Node()
			lefttree = buildTree(node.left,leftRDD,filterList.add(bestSplit))
			righttree = buildTree(node.right,rightRDD,filterList.add(bestSplit.invert()))
		}
		node
	}

###Testing

#####Unit testing
As a standard programming practice, unit tests will be written to test the important building blocks.

####Comparison with other libraries
There are several machine learning libraries in other languages. The scikit-learn library will be used a benchmark for functional tests.

###Constraints
+ Two class labels -- The first implementation will support only binary labels.
+ Class weights -- Class weighting option (useful for highly unblanaced data) will not be supported
+ Sanity checks -- The input data sanity checks will not be performed. Ideally, a separate pre-processing step (that that is common to all ML algorithms) should handle this.

## Future Work
+ Weights to handle unbalanced classes
+ Ensemble methods -- random forests, boosting, etc.

##References
1. Hastie, Tibshirani, Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer 2009.
2. Biswanath, Herbach, Basu and Roberto. PLANET: Massively Parallel Learning of Tree Ensembles with MapReduce, VLDB 2009.