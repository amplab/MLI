/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.tree
import javax.naming.OperationNotSupportedException
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.SparkContext
import org.apache.spark.util.StatCounter
import org.apache.spark.Logging
import ml.tree.impurity.{Variance, Entropy, Gini, Impurity}
import ml.tree.node.{Prediction, NodeStats, NodeModel, Node}
import ml.tree.strategy.Strategy
import ml.tree.split.{SplitPredicate, Split}
import org.apache.spark.broadcast.Broadcast


/*
 * Class for building the Decision Tree model. Should be used for both classification and regression tree.
 */
class DecisionTree (
  val input: RDD[(Double, Array[Double])], //input RDD
  val maxDepth: Int, // depth of the tree
  val numSplitPredicates: Int, // number of bins per features
  val fraction: Double, // fraction of the data to be used for performing quantile calculation
  val strategy: Strategy, // classification or regression
  val impurity: Impurity, // impurity calculation strategy (variance, gini, entropy, etc.)
  val sparkContext : SparkContext) { 

  //Calculating length of the features
  val featureLength = input.first._2.length
  println("feature length = " + featureLength)
  
  //Sampling a fraction of the input RDD
  val sampledData = input.sample(false, fraction, 42).cache()
  
  //Sorting the sampled data along each feature and storing it for quantile calculation
  val sortedSampledFeatures = {
    val sortedFeatureArray = new Array[RDD[Double]](featureLength)
    0 until featureLength foreach {
      i => sortedFeatureArray(i) = sampledData.map(x => x._2(i) -> None).sortByKey(true).map(_._1)
    }
    sortedFeatureArray
  }

  val numSamples = sampledData.count
  println("num samples = " + numSamples)

  // Calculating the index to jump to find the quantile points
  val stride = scala.math.max(numSamples / numSplitPredicates, 1)
  println("stride = " + stride)

    //Calculating all possible splits for the features
  val allSplitsList = for {
    featureIndex <- 0 until featureLength;
    index <- stride until numSamples - 1 by stride
  } yield createSplit(featureIndex, index)

  //Remove duplicate splits. Especially help for one-hot encoded categorical variables.
  val allSplits = sparkContext.broadcast(allSplitsList.toSet)

  //for (split <- allSplits) yield println(split)

  /*
   * Find the exact value using feature index and index into the sorted features
   */
  def valueAtRDDIndex(featuresIndex: Long, index: Long): Double = {
    sortedSampledFeatures(featuresIndex.toInt).collect()(index.toInt)
  }

  /*
   * Create splits using feature index and index into the sorted features
   */
  def createSplit(featureIndex: Int, index: Long): Split = {
    new Split(featureIndex, valueAtRDDIndex(featureIndex, index))
  }

  /*
   * Empty Node class used to terminate leaf nodes
   */
  private class LeafNode(val data: RDD[(Double, Array[Double])]) extends Node {
    def isLeaf = true
    def left = throw new OperationNotSupportedException("EmptyNode.left")
    def right = throw new OperationNotSupportedException("EmptyNode.right")
    def depth = throw new OperationNotSupportedException("EmptyNode.depth")
    def splitPredicates = throw new OperationNotSupportedException("EmptyNode.splitPredicates")
    def splitPredicate = throw new OperationNotSupportedException("EmptyNode.splitPredicate")
    override def toString() = "Empty"
    val prediction: Prediction = {
      val countZero: Double = data.filter(x => (x._1 == 0.0)).count
      val countOne: Double = data.filter(x => (x._1 == 1.0)).count
      val countTotal: Double = countZero + countOne
      new Prediction(countOne / countTotal, Map(0.0 -> countZero, 1.0 -> countOne))
    }
  }

  /*
   * Top node for building a classification tree
   */
  private class TopClassificationNode extends ClassificationNode(input.cache, 0, List[SplitPredicate](), new NodeStats) {
    override def toString() = "[" + left + "[" + "TopNode" + "]" + right + "]"
  }

  /*
   * Class for each node in the classification tree
   */
  private class ClassificationNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats : NodeStats) 
  extends DecisionNode(data, depth, splitPredicates, nodeStats) {

    // Prediction at each classification node
    val prediction: Prediction = {
      val countZero: Double = data.filter(x => (x._1 == 0.0)).count
      val countOne: Double = data.filter(x => (x._1 == 1.0)).count
      val countTotal: Double = countZero + countOne
      new Prediction(countOne / countTotal, Map(0.0 -> countZero, 1.0 -> countOne))
    }

    //Static factory method. Put it in a better location.
    def createNode(anyData: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats : NodeStats) = new ClassificationNode(anyData, depth, splitPredicates, nodeStats)

  }

  /*
   * Top node for building a regression tree
   */
  private class TopRegressionNode(nodeStats : NodeStats) extends RegressionNode(input.cache, 0, List[SplitPredicate](), nodeStats) {
    override def toString() = "[" + left + "[" + "TopNode" + "]" + right + "]"
  }

  /*
   * Class for each node in the regression tree
   */
  private class RegressionNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats : NodeStats) 
  extends DecisionNode(data, depth, splitPredicates, nodeStats) {
    
    // Prediction at each regression node
    val prediction: Prediction = new Prediction(data.map(_._1).mean, Map())
    
    //Static factory method. Put it in a better location.
    def createNode(anyData: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats : NodeStats) = new RegressionNode(anyData, depth, splitPredicates, nodeStats)
  }

  abstract class DecisionNode(
      val data: RDD[(Double, Array[Double])],
      val depth: Int,
      val splitPredicates: List[SplitPredicate],
      val nodeStats : NodeStats) extends Node {

    //TODO: Change empty logic
    val splits = splitPredicates.map(x => x.split)

    //TODO: Think about the merits of doing BFS and removing the parents RDDs from memory instead of doing DFS like below.
    val (left, right, splitPredicate, isLeaf) = createLeftRightChild()
    override def toString() = "[" + left + "[" + this.splitPredicate + " prediction = " + this.prediction + "]" + right + "]"
    def createNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats : NodeStats): DecisionNode
    def createLeftRightChild(): (Node, Node, Option[SplitPredicate], Boolean) = {
      if (depth > maxDepth) {
        (new LeafNode(data), new LeafNode(data), None, true)
      } else {
        println("split count " + splits.length)
        val split_gain = findBestSplit(nodeStats)
        val (split, gain, leftNodeStats, rightNodeStats) = split_gain
        println("Selected split = " + split + " with gain = " + gain, "left node stats = " + leftNodeStats + " right node stats = " + rightNodeStats)
        if (split_gain._2 > 0) {
          println("creating new nodes at depth = " + depth)
          val leftPredicate = new SplitPredicate(split, true)
          val rightPredicate = new SplitPredicate(split, false)
          val leftData = data.filter(sample => sample._2(leftPredicate.split.feature) <= leftPredicate.split.threshold).cache
          val rightData = data.filter(sample => sample._2(rightPredicate.split.feature) > rightPredicate.split.threshold).cache
          val leftNode = if (leftData.count != 0) createNode(leftData, depth + 1, splitPredicates ::: List(leftPredicate), leftNodeStats) else new LeafNode(data)
          val rightNode = if (rightData.count != 0) createNode(rightData, depth + 1, splitPredicates ::: List(rightPredicate), rightNodeStats) else new LeafNode(data)
          (leftNode, rightNode, Some(leftPredicate), false)
        } else {
          println("not creating more child nodes since gain is not greater than 0")
          (new LeafNode(data), new LeafNode(data), None, true)
        }
      }
    }

    def findBestSplit(nodeStats: NodeStats): (Split, Double, NodeStats, NodeStats) = {

      //TODO: Also remove splits that are subsets of previous splits
      val availableSplits = allSplits.value filterNot (split => splits contains split)
      println("availableSplit count " + availableSplits.size)
      //availableSplits.map(split1 => (split1, impurity.calculateGain(split1, data))).reduce(comparePair(_, _))

      strategy match {
        case Strategy("Classification") => {

          val splitWiseCalculations = data.flatMap(sample => {
            val label = sample._1
            val features = sample._2
            val leftOrRight = for {
              split <- availableSplits.toSeq
              featureIndex = split.feature
              threshold = split.threshold
            } yield { if (features(featureIndex) <= threshold) (split, "left", label) else (split, "right", label) }
            leftOrRight
          }).map(k => (k, 1))

          val gainCalculations = splitWiseCalculations.countByKey()
          	.toMap //TODO: Hack to go from mutable to immutable map. Clean this up if needed.

          val split_gain_list = for (
            split <- availableSplits;
            gain = impurity.calculateClassificationGain(split, gainCalculations)
          ) yield (split, gain)

          val split_gain = split_gain_list.reduce(comparePair(_, _))
          (split_gain._1, split_gain._2, new NodeStats, new NodeStats)

        }
        case Strategy("Regression") => {

          val splitWiseCalculations = data.flatMap(sample => {
            val label = sample._1
            val features = sample._2
            val leftOrRight = for {
              split <- availableSplits.toSeq
              featureIndex = split.feature
              threshold = split.threshold
            } yield {if (features(featureIndex) <= threshold) ((split, "left"), label) else ((split, "right"), label)}
            leftOrRight
          })

          // Calculate variance for each split
          val splitVariancePairs = splitWiseCalculations.groupByKey().map(x => x._1 -> calculateVarianceSize(x._2)).collect
          //Tuple array to map conversion
          val gainCalculations = scala.collection.immutable.Map(splitVariancePairs: _*)

          val split_gain_list = for (
            split <- availableSplits;
            (gain, leftNodeStats, rightNodeStats) = impurity.calculateRegressionGain(split, gainCalculations, nodeStats)
          ) yield (split, gain, leftNodeStats, rightNodeStats)

          val split_gain = split_gain_list.reduce(compareRegressionPair(_, _))
          (split_gain._1, split_gain._2,split_gain._3, split_gain._4)
        }
      }
    }

    def calculateVarianceSize(seq: Seq[Double]): (Double, Double, Long) = {
      val stat = StatCounter(seq)
      (stat.mean, stat.variance, stat.count)
    }


  }


  def comparePair(x: (Split, Double), y: (Split, Double)): (Split, Double) = {
    if (x._2 > y._2) x else y
  }

  def compareRegressionPair(x: (Split, Double, NodeStats, NodeStats), y: (Split, Double, NodeStats, NodeStats)): (Split, Double, NodeStats, NodeStats) = {
    if (x._2 > y._2) x else y
  }


  def buildTree(): Node = {
    strategy match {
      case Strategy("Classification") => new TopClassificationNode()
      case Strategy("Regression")     => {
        val count = input.count
        //TODO: calculate mean and variance together
        val variance = input.map(x => x._1).variance
        val mean = input.map(x => x._1).mean
        val nodeStats = new NodeStats(count = Some(count), variance = Some(variance), mean = Some(mean))
        new TopRegressionNode(nodeStats)
      }
    }
  }

}


object DecisionTree {
  def train(
    input: RDD[(Double, Array[Double])],
    numSplitPredicates: Int,
    strategy: Strategy,
    impurity: Impurity, 
    maxDepth : Int, 
    fraction : Double,
    sparkContext : SparkContext): Option[NodeModel] = {
    new DecisionTree(
      input = input,
      numSplitPredicates = numSplitPredicates,
      strategy = strategy,
      impurity = impurity,
      maxDepth = maxDepth,
      fraction = fraction,
      sparkContext = sparkContext)
      .buildTree
      .extractModel
  }
}







