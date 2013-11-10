package ml.tree.node

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import ml.tree.split.{Split, SplitPredicate}
import org.apache.spark.broadcast.Broadcast
import ml.tree.impurity.Impurity
import ml.tree.strategy.Strategy
import org.apache.spark.util.StatCounter
import javax.naming.OperationNotSupportedException
import ml.tree.Metrics._
import scala.Some
import ml.tree.strategy.Strategy
import ml.tree.split.Split
import scala.collection.mutable

abstract class DecisionNode(
                             val data: RDD[(Double, Array[Double])],
                             val depth: Int,
                             val splitPredicates: List[SplitPredicate],
                             val nodeStats: NodeStats,
                             val allSplits: Broadcast[Set[Split]],
                             val impurity: Impurity,
                             val strategy: Strategy,
                             val maxDepth: Int) extends Node {

  //TODO: Change empty logic
  val splits = splitPredicates.map(x => x.split)

  //TODO: Think about the merits of doing BFS and removing the parents RDDs from memory instead of doing DFS like below.
  val (left, right, splitPredicate, isLeaf) = createLeftRightChild()

  override def toString() = "[" + left + "[" + this.splitPredicate + " prediction = " + this.prediction + "]" + right + "]"

  def createNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats: NodeStats): DecisionNode

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

  def comparePair(x: (Split, Double), y: (Split, Double)): (Split, Double) = {
    if (x._2 > y._2) x else y
  }

  def compareRegressionPair(x: (Split, Double, NodeStats, NodeStats), y: (Split, Double, NodeStats, NodeStats)): (Split, Double, NodeStats, NodeStats) = {
    if (x._2 > y._2) x else y
  }


  def findBestSplit(nodeStats: NodeStats): (Split, Double, NodeStats, NodeStats) = {

    //TODO: Also remove splits that are subsets of previous splits
    val availableSplits = allSplits.value filterNot (split => splits contains split)
    println("availableSplit count " + availableSplits.size)
    //availableSplits.map(split1 => (split1, impurity.calculateGain(split1, data))).reduce(comparePair(_, _))

    strategy match {
      case Strategy("Classification") => {

        //Write a function that takes an RDD and list of splits
        //and returns a map of (split, <left/right>, label) -> count

        val splits = availableSplits.toSeq

        //Modify numLabels to support multiple classes in the future
        val numLabels = 2
        val numChildren = 2
        val lenSplits = splits.length
        val outputVectorLength = numLabels * numChildren * lenSplits
        val vecToVec : RDD[Array[Long]] = data.map(
          sample => {
            val storage : Array[Long] = new Array[Long](outputVectorLength)
            val label = sample._1
            val features = sample._2
            splits.zipWithIndex.foreach{case (split, i) =>
              val featureIndex = split.feature
              val threshold = split.threshold
              if (features(featureIndex) <= threshold) { //left node
                val index = i*(numLabels*numChildren) + label.toInt
                storage(index) = 1
              } else{ //right node
                val index = i*(numLabels*numChildren) + numLabels + label.toInt
                storage(index) = 1
              }
            }
            storage
          }
        )

        val countVecToVec : Array[Long] = vecToVec.reduce((a1,a2) => NodeHelper.sumTwoArrays(a1,a2))

        //TOOD: Unnecessary step. Use indices directly instead of creating a map. Not a big hit in performance. Optimize later.
        var newGainCalculations = Map[(Split,String,Double),Long]()
        splits.zipWithIndex.foreach{case(split,i) =>
            newGainCalculations += ((split,"left",0.0) -> countVecToVec(i*(numLabels*numChildren) + 0))
            newGainCalculations += ((split,"left",1.0) -> countVecToVec(i*(numLabels*numChildren) + 1))
            newGainCalculations += ((split,"right",0.0) -> countVecToVec(i*(numLabels*numChildren) + numLabels + 0))
            newGainCalculations += ((split,"right",1.0) -> countVecToVec(i*(numLabels*numChildren) + numLabels + 1))
         }

        //TODO: Vectorize this operation as well
        val split_gain_list = for (
          split <- availableSplits;
          //gain = impurity.calculateClassificationGain(split, gainCalculations)
          gain = impurity.calculateClassificationGain(split, newGainCalculations)
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
          } yield {
            if (features(featureIndex) <= threshold) ((split, "left"), label) else ((split, "right"), label)
          }
          leftOrRight
        })

        // Calculate variance for each split
        val splitVariancePairs = splitWiseCalculations
          .groupByKey()
          .map(x => x._1 -> {val stat = StatCounter(x._2); (stat.mean, stat.variance, stat.count)})
          .collect
        //Tuple array to map conversion
        val gainCalculations = scala.collection.immutable.Map(splitVariancePairs: _*)

        val split_gain_list = for (
          split <- availableSplits;
          (gain, leftNodeStats, rightNodeStats) = impurity.calculateRegressionGain(split, gainCalculations, nodeStats)
        ) yield (split, gain, leftNodeStats, rightNodeStats)

        val split_gain = split_gain_list.reduce(compareRegressionPair(_, _))
        (split_gain._1, split_gain._2, split_gain._3, split_gain._4)
      }
    }
  }

  def calculateVarianceSize(seq: Seq[Double]): (Double, Double, Long) = {
    val stat = StatCounter(seq)
    (stat.mean, stat.variance, stat.count)
  }

}


/*
 * Top node for building a classification tree
 */
class TopClassificationNode(input: RDD[(Double, Array[Double])], allSplits: Broadcast[Set[Split]], impurity: Impurity, strategy: Strategy, maxDepth: Int) extends ClassificationNode(input.cache, 1, List[SplitPredicate](), new NodeStats, allSplits, impurity, strategy, maxDepth) {
  override def toString() = "[" + left + "[" + "TopNode" + "]" + right + "]"
}

/*
 * Class for each node in the classification tree
 */
class ClassificationNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats: NodeStats, allSplits: Broadcast[Set[Split]], impurity: Impurity, strategy: Strategy, maxDepth: Int)
  extends DecisionNode(data, depth, splitPredicates, nodeStats, allSplits, impurity, strategy, maxDepth) {

  // Prediction at each classification node
  val prediction: Prediction = {
    val countZero: Double = data.filter(x => (x._1 == 0.0)).count
    val countOne: Double = data.filter(x => (x._1 == 1.0)).count
    val countTotal: Double = countZero + countOne
    new Prediction(countOne / countTotal, Map(0.0 -> countZero, 1.0 -> countOne))
  }

  //Static factory method. Put it in a better location.
  def createNode(anyData: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats: NodeStats)
  = new ClassificationNode(anyData, depth, splitPredicates, nodeStats, allSplits, impurity, strategy, maxDepth)

}

/*
 * Top node for building a regression tree
 */
class TopRegressionNode(input: RDD[(Double, Array[Double])], nodeStats: NodeStats, allSplits: Broadcast[Set[Split]], impurity: Impurity, strategy: Strategy, maxDepth: Int) extends RegressionNode(input.cache, 1, List[SplitPredicate](), nodeStats, allSplits, impurity, strategy, maxDepth) {
  override def toString() = "[" + left + "[" + "TopNode" + "]" + right + "]"
}

/*
 * Class for each node in the regression tree
 */
class RegressionNode(data: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats: NodeStats, allSplits: Broadcast[Set[Split]], impurity: Impurity, strategy: Strategy, maxDepth: Int)
  extends DecisionNode(data, depth, splitPredicates, nodeStats, allSplits, impurity, strategy, maxDepth) {

  // Prediction at each regression node
  val prediction: Prediction = new Prediction(data.map(_._1).mean, Map())

  //Static factory method. Put it in a better location.
  def createNode(anyData: RDD[(Double, Array[Double])], depth: Int, splitPredicates: List[SplitPredicate], nodeStats: NodeStats)
  = new RegressionNode(anyData, depth, splitPredicates, nodeStats, allSplits, impurity, strategy, maxDepth)
}

/*
 * Empty Node class used to terminate leaf nodes
 */
class LeafNode(val data: RDD[(Double, Array[Double])]) extends Node {
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

object NodeHelper extends Serializable {

  //There definitely has to be a library function to do this!
  def sumTwoArrays(a1 : Array[Long], a2 : Array[Long]) : Array[Long] = {
    val storage = new Array[Long](a1.length)
    for (i <- 0 until a1.length){storage(i) = a1(i) + a2(i)}
    storage
  }

}



