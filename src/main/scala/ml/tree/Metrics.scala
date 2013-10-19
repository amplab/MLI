package ml.tree

import org.apache.spark.SparkContext._
import ml.tree.node.NodeModel
import org.apache.spark.rdd.RDD

/*
Helper methods for measuring performance of ML algorithms
 */
object Metrics {

  //TODO: Make these generic MLTable metrics.
  def accuracyScore(tree : Option[NodeModel], data : RDD[(Double, Array[Double])]) : Double = {
    if (tree.isEmpty) return 1 //TODO: Throw exception
    val correctCount = data.filter(y => tree.get.predict(y._2) == y._1).count()
    val count = data.count()
    print("correct count = " +  correctCount)
    print("training data count = " + count)
    correctCount.toDouble / count
  }

  //TODO: Make these generic MLTable metrics
  def meanSquaredError(tree : Option[NodeModel], data : RDD[(Double, Array[Double])]) : Double = {
    if (tree.isEmpty) return 1 //TODO: Throw exception
    val meanSumOfSquares = data.map(y => (tree.get.predict(y._2) - y._1)*(tree.get.predict(y._2) - y._1)).mean()
    print("meanSumOfSquares = " + meanSumOfSquares)
    meanSumOfSquares
  }


}
