package ml.tree.node

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.rdd.RDD
import ml.tree.split.SplitPredicate
import ml.tree.Metrics._

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
