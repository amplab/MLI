package mli.ml.clustering

import mli.interface._
import mli.ml._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.regression.LabeledPoint

class KMeansModel(
                trainingTbl: MLTable,
                trainingParams: KMeansParameters,
                trainingTime: Long,
                val model: org.apache.spark.mllib.clustering.KMeansModel)
  extends Model[KMeansParameters](trainingTbl, trainingTime, trainingParams) {


  /* Predicts the label of a given data point. */
  def predict(x: MLRow) : MLValue = {
    MLValue(model.predict(x.toDoubleArray))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain() : String = {
    "Centers: " + model.clusterCenters.mkString(" ")
  }
}

case class KMeansParameters(
                          k: Int = 2,
                          maxIterations: Int = 100,
                          runs: Int = 1,
                          initializationMode: String = KMeans.RANDOM,
                          epsilon: Double = 1e-4)
  extends AlgorithmParameters


object KMeansAlgorithm extends Algorithm[KMeansParameters] {

  def defaultParameters() = KMeansParameters()

  def train(data: MLTable, params: KMeansParameters): KMeansModel = {

    // Initialization
    assert(data.numRows > 0)
    assert(data.numCols > 1)

    val startTime = System.currentTimeMillis

    //Run the K-Means algorithm on the data.
    val model = KMeans.train(
      data.toDoubleArrayRDD().cache(),
      params.k,
      params.maxIterations,
      params.runs,
      params.initializationMode)

    val trainTime = System.currentTimeMillis - startTime

    new KMeansModel(data, params, trainTime, model)
  }

}






