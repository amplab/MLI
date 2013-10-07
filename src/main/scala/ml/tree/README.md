#Tree design doc
Decision tree classifiers are both popular supervised learning algorithms and also building blocks for other ensemble learning algorithms such as random forests, boosting, etc. This document discusses its implementation in the Spark project.

#Usage
DecisionTreeRunner <master>[slices] --strategy <Classification,Regression> --trainDataDir path --testDataDir path [--maxDepth num] [--impurity <Gini,Entropy,Variance>] [--samplingFractionForSplitCalculation num] 
  

#Example
sbt/sbt "run-main ml.tree.TreeRunner local[2] --strategy Classification --trainDataDir ../train_data --testDataDir ../test_data --maxDepth 1 --impurity Gini --samplingFractionForSplitCalculation 1"

This command will create a decision tree model using the training data in the *trainDataDir* and calculate test error using the data in the *testDataDir*. The mis-classification error is calculated for a Classification *strategy* and mean squared error is calculated for the Regression *strategy*.