package ml.tree

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

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
