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


/*
 * Abstract Node class as a template for implementing various types of nodes in the decision tree.
 */
abstract class Node {
  //Method for checking whether the class has any left/right child nodes.
  def isLeaf: Boolean
  //Left/Right child nodes
  def left: Node
  def right: Node
  //Depth of the node from the top node
  def depth: Int
  //RDD data as an input to the node
  def data: RDD[(Double, Array[Double])]
  //List of split predicates applied to the base RDD thus far
  def splitPredicates: List[SplitPredicate]
  // Split to arrive at the node
  def splitPredicate: Option[SplitPredicate]
  //Extract model
  def extractModel: Option[NodeModel] = {
    //Add probability logic
    if (!splitPredicate.isEmpty) { Some(new NodeModel(splitPredicate, left.extractModel, right.extractModel, depth, isLeaf, Some(prediction))) }
    else {
      // Using -1 as depth
      Some(new NodeModel(None, None, None, depth, isLeaf, Some(prediction)))
    }
  }
  def prediction: Prediction
}

/**
 * The decision tree model class that
 */
class NodeModel(
  val splitPredicate: Option[SplitPredicate],
  val trueNode: Option[NodeModel],
  val falseNode: Option[NodeModel],
  val depth: Int,
  val isLeaf: Boolean,
  val prediction: Option[Prediction]) extends ClassificationModel {

  override def toString() = if (!splitPredicate.isEmpty) {
    "[" + trueNode.get + "\n" + "[" + "depth = " + depth + ", split predicate = " + this.splitPredicate.get + ", predict = " + this.prediction + "]" + "]\n" + falseNode.get
  } else {
    "Leaf : " + "depth = " + depth + ", predict = " + prediction + ", isLeaf = " + isLeaf
  }

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return RDD[Int] where each entry contains the corresponding prediction
   */
  def predict(testData: RDD[Array[Double]]): RDD[Double] = {
    testData.map { x => predict(x) }
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return Int prediction from the trained model
   */
  def predict(testData: Array[Double]): Double = {
    //TODO: Modify this logic to handle regression
    val pred = prediction.get
    if (this.isLeaf) {
      if (pred.prob > 0.5) 1 else 0
    } else {
      val spPred = splitPredicate.get
      if (testData(spPred.split.feature) <= spPred.split.threshold) {
        trueNode.get.predict(testData)
      } else {
        falseNode.get.predict(testData)
      }
    }
  }

}

/*
 * Class used to store the prediction values at each node of the tree.
 */
class Prediction(val prob: Double, val distribution: Map[Double, Double]) extends Serializable	{
  override def toString = { "probability = " + prob + ", distribution = " + distribution }
}

/*
 * Class for storing splits -- feature index and threshold
 */
case class Split(val feature: Int, val threshold: Double) {
  override def toString = "feature = " + feature + ", threshold = " + threshold
}

/*
 * Class for storing the split predicate.
 */
class SplitPredicate(val split: Split, lessThanEqualTo: Boolean = true) extends Serializable {
  override def toString = "split = " + split.toString + ", lessThan = " + lessThanEqualTo
}

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
          val splitVariancePairs = splitWiseCalculations.groupByKey().map(x => x._1 -> ParVariance.calculateVarianceSize(x._2)).collect
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

object ParVariance extends Serializable {
  
    def calculateVarianceSize(seq: Seq[Double]): (Double, Double, Long) = {
    val stat = StatCounter(seq)
    (stat.mean, stat.variance, stat.count)
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

case class Strategy(val name: String)

class NodeStats(
  val gini: Option[Double] = None,
  val entropy: Option[Double] = None,
  val mean: Option[Double] = None,
  val variance: Option[Double] = None,
  val count: Option[Long] = None) extends Serializable{
  override def toString = "variance = " + variance + "count = " + count + "mean = " + mean
}


trait Impurity {

  def calculateClassificationGain(split: Split, calculations : Map[(Split, String, Double),Long]): Double = {
          val leftRddZeroCount = calculations.getOrElse((split,"left",0.0),0L).toDouble;
          val rightRddZeroCount = calculations.getOrElse((split,"right",0.0),0L).toDouble;
          val leftRddOneCount = calculations.getOrElse((split,"left",1.0),0L).toDouble;
          val rightRddOneCount = calculations.getOrElse((split,"right",1.0),0L).toDouble;
          val leftRddCount = leftRddZeroCount + leftRddOneCount;
          val rightRddCount = rightRddZeroCount + rightRddOneCount;
          val totalZeroCount = leftRddZeroCount + rightRddZeroCount;
          val totalOneCount = leftRddOneCount + rightRddOneCount;
          val totalCount = totalZeroCount + totalOneCount;
          val gain = {
            if (leftRddCount == 0 || rightRddCount == 0) 0
            else {
              val topGini = calculate(totalZeroCount,totalOneCount)
              val leftWeight = leftRddCount / totalCount
              val leftGini = calculate(leftRddZeroCount,leftRddOneCount) * leftWeight
              val rightWeight = rightRddCount / totalCount
              val rightGini = calculate(rightRddZeroCount,rightRddOneCount) * rightWeight
              topGini - leftGini - rightGini
            }
          }
          gain
  }
  
  def calculateRegressionGain(split: Split, calculations : Map[(Split, String),(Double, Double, Long)], nodeStats : NodeStats): (Double, NodeStats, NodeStats) = {
    val topCount = nodeStats.count.get
    val leftCount = calculations.getOrElse((split,"left"),(0,0,0L))._3
    val rightCount = calculations.getOrElse((split,"right"),(0,0,0L))._3
    if (leftCount == 0 || rightCount == 0){
    	// No gain return values
    	//println("leftCount = " + leftCount + "rightCount = " + rightCount + " topCount = " + topCount)
          (0, new NodeStats, new NodeStats)
    } else{
    	val topVariance = nodeStats.variance.get
    	val leftMean = calculations((split,"left"))._1
    	val leftVariance = calculations((split,"left"))._2
    	val rightMean = calculations((split,"right"))._1
    	val rightVariance = calculations((split,"right"))._2
    	//TODO: Check and if needed improve these toDouble conversions
    	val gain = topVariance - ((leftCount.toDouble / topCount) * leftVariance) - ((rightCount.toDouble/topCount) * rightVariance)
    	(gain, 
    	    new NodeStats(mean = Some(leftMean), variance = Some(leftVariance), count = Some(leftCount)), 
    	    new NodeStats(mean = Some(rightMean), variance = Some(rightVariance), count = Some(rightCount)))
    }
  }
  
  def calculate(c0 : Double, c1 : Double): Double

}


object Gini extends Impurity {

  def calculate(c0 : Double, c1 : Double): Double = {
    val total = c0 + c1
    val f0 = c0 / total
    val f1 = c1 / total
    1 - f0*f0 - f1*f1
  }

}

object Entropy extends Impurity {

  def log2(x: Double) = scala.math.log(x) / scala.math.log(2)

  def calculate(c0: Double, c1: Double): Double = {
    if (c0 == 0 || c1 == 0) {
      0
    } else {
      val total = c0 + c1
      val f0 = c0 / total
      val f1 = c1 / total
      -(f0 * log2(f0)) - (f1 * log2(f1))
    }
  }

}

object Variance extends Impurity {
  def calculate(c0: Double, c1: Double): Double = throw new OperationNotSupportedException("Variance.calculate")
}

object TreeRunner extends Logging {
  val usage = """
    Usage: DecisionTreeRunner <master>[slices] --strategy <Classification,Regression> --trainDataDir path --testDataDir path [--maxDepth num] [--impurity <Gini,Entropy,Variance>] [--samplingFractionForSplitCalculation num] 
  """
    
  def main(args: Array[String]) {

    if (args.length < 2) {
		  System.err.println(usage)
		  System.exit(1)
	  }
    
    /**START Experimental*/
    System.setProperty("spark.cores.max", "8")
    /**END Experimental*/
    val sc = new SparkContext(args(0), "Decision Tree Runner",
      System.getenv("SPARK_HOME"), Seq(System.getenv("SPARK_EXAMPLES_JAR")))


    val arglist = args.toList.drop(1)
    type OptionMap = Map[Symbol, Any]

    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map
        case "--strategy" :: string :: tail => nextOption(map ++ Map('strategy -> string), tail)
        case "--trainDataDir" :: string :: tail => nextOption(map ++ Map('trainDataDir -> string), tail)
        case "--testDataDir" :: string :: tail => nextOption(map ++ Map('testDataDir -> string), tail)
        case "--impurity" :: string :: tail => nextOption(map ++ Map('impurity -> string), tail)
        case "--maxDepth" :: string :: tail => nextOption(map ++ Map('maxDepth -> string), tail)
        case "--samplingFractionForSplitCalculation" :: string :: tail => nextOption(map ++ Map('samplingFractionForSplitCalculation -> string), tail)
        case string :: Nil =>  nextOption(map ++ Map('infile -> string), list.tail)
        case option :: tail => println("Unknown option "+option) 
                               exit(1) 
      }
    }
    val options = nextOption(Map(),arglist)
    println(options)
    //TODO: Add check for acceptable string inputs
    
    val trainData = TreeUtils.loadLabeledData(sc, options.get('trainDataDir).get.toString)
    val strategyStr = options.get('strategy).get.toString
    val impurityStr = options.getOrElse('impurity,"Gini").toString
    val impurity = {
    	impurityStr match {
    	  case "Gini" => Gini
    	  case "Entropy" => Entropy
    	  case "Variance" => Variance
    	}
    }
    val maxDepth = options.getOrElse('maxDepth,"1").toString.toInt
    val fraction = options.getOrElse('samplingFractionForSplitCalculation,"1.0").toString.toDouble
    
    val tree = DecisionTree.train(
      input = trainData,
      numSplitPredicates = 1000,
      strategy = new Strategy(strategyStr),
      impurity = impurity,
      maxDepth = maxDepth,
      fraction = fraction,
      sparkContext = sc)
    println(tree)
    //println("prediction = " + tree.get.predict(Array(1.0, 2.0)))
    
    val testData = TreeUtils.loadLabeledData(sc, options.get('testDataDir).get.toString)

    
    val testError = {
      strategyStr match {
        case "Classification" => accuracyScore(tree, testData)
        case "Regression" => meanSquaredError(tree, testData)
      }
    }
    print("error = " + testError)
    
  }
  
  def accuracyScore(tree : Option[ml.tree.NodeModel], data : RDD[(Double, Array[Double])]) : Double = {
    if (tree.isEmpty) return 1 //TODO: Throw exception
    val correctCount = data.filter(y => tree.get.predict(y._2) == y._1).count()
    val count = data.count()
    print("correct count = " +  correctCount)
    print("training data count = " + count)
    correctCount.toDouble / count
  }

  def meanSquaredError(tree : Option[ml.tree.NodeModel], data : RDD[(Double, Array[Double])]) : Double = {
    if (tree.isEmpty) return 1 //TODO: Throw exception
    val meanSumOfSquares = data.map(y => (tree.get.predict(y._2) - y._1)*(tree.get.predict(y._2) - y._1)).mean
    print("meanSumOfSquares = " + meanSumOfSquares)
    meanSumOfSquares
  }

    
}


/**
 * Helper methods to load and save data
 * Data format:
 * <l>, <f1> <f2> ...
 * where <f1>, <f2> are feature values in Double and <l> is the corresponding label as Double.
 */
object TreeUtils {

  /**
   * @param sc SparkContext
   * @param dir Directory to the input data files.
   * @return An RDD of tuples. For each tuple, the first element is the label, and the second
   *         element represents the feature values (an array of Double).
   */
  def loadLabeledData(sc: SparkContext, dir: String): RDD[(Double, Array[Double])] = {
    sc.textFile(dir).map { line =>
      val parts = line.trim().split(",")
      val label = parts(0).toDouble
      val features = parts.slice(1,parts.length).map(_.toDouble)
      //val features = parts.slice(1, 30).map(_.toDouble)
      (label, features)
    }
  }

  def saveLabeledData(data: RDD[(Double, Array[Double])], dir: String) {
    val dataStr = data.map(x => x._1 + "," + x._2.mkString(" "))
    dataStr.saveAsTextFile(dir)
  }

}
