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
import ml.tree.strategy.Strategy
import ml.tree.split.{SplitPredicate, Split}
import org.apache.spark.broadcast.Broadcast
import scala.Some
import ml.tree.strategy.Strategy
import ml.tree.split.Split
import ml.tree.node._
import ml.tree.Metrics._
import scala.Some
import ml.tree.strategy.Strategy
import ml.tree.split.Split


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
  println("sampled data size for quantile calculation = " + sampledData.count)

  //Sorting the sampled data along each feature and storing it for quantile calculation
  println("started sorting sampled data")
  val sortedSampledFeatures = {
    val sortedFeatureArray = new Array[Array[Double]](featureLength)
    0 until featureLength foreach {
      i => sortedFeatureArray(i) = sampledData.map(x => x._2(i) -> None).sortByKey(true).map(_._1).collect()
    }
    sortedFeatureArray
  }
  println("finished sorting sampled data")

  val numSamples = sampledData.count
  println("num samples = " + numSamples)

  // Calculating the index to jump to find the quantile points
  val stride = scala.math.max(numSamples / numSplitPredicates, 1)
  println("stride = " + stride)

  //Calculating all possible splits for the features
  println("calculating all possible splits for features")
  val allSplitsList = for {
    featureIndex <- 0 until featureLength;
    index <- stride until numSamples - 1 by stride
  } yield createSplit(featureIndex, index)
  println("finished calculating all possible splits for features")

  //Remove duplicate splits. Especially help for one-hot encoded categorical variables.
  val allSplits = sparkContext.broadcast(allSplitsList.toSet)

  //for (split <- allSplits) yield println(split)

  /*
   * Find the exact value using feature index and index into the sorted features
   */
  def valueAtRDDIndex(featuresIndex: Long, index: Long): Double = {
    sortedSampledFeatures(featuresIndex.toInt)(index.toInt)
  }

  /*
   * Create splits using feature index and index into the sorted features
   */
  def createSplit(featureIndex: Int, index: Long): Split = {
    new Split(featureIndex, valueAtRDDIndex(featureIndex, index))
  }

  def buildTree(): Node = {

    println("building decision tree")

    strategy match {
      case Strategy("Classification") => new TopClassificationNode(input, allSplits, impurity, strategy, maxDepth)
      case Strategy("Regression")     => {
        val count = input.count
        //TODO: calculate mean and variance together
        val variance = input.map(x => x._1).variance
        val mean = input.map(x => x._1).mean
        val nodeStats = new NodeStats(count = Some(count), variance = Some(variance), mean = Some(mean))
        new TopRegressionNode(input, nodeStats,allSplits, impurity, strategy, maxDepth)
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
    val tree = new DecisionTree(
      input = input,
      numSplitPredicates = numSplitPredicates,
      strategy = strategy,
      impurity = impurity,
      maxDepth = maxDepth,
      fraction = fraction,
      sparkContext = sparkContext)
      .buildTree
      .extractModel

    println("calculating performance on training data")
    val trainingError = {
      strategy match {
        case Strategy("Classification") => accuracyScore(tree, input)
        case Strategy("Regression") => meanSquaredError(tree, input)
      }
    }
    println("accuracy = " + trainingError)

    tree
  }
}