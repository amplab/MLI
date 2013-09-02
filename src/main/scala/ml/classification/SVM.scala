package mli.ml.classification

import mli.interface._
import mli.ml._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint

class SVMModel(
    trainingTbl: MLTable,
    trainingParams: SVMParameters,
    trainingTime: Long,
    val model: org.apache.spark.mllib.classification.SVMModel)
  extends Model[SVMParameters](trainingTbl, trainingTime, trainingParams) {


  /* Predicts the label of a given data point. */
  def predict(x: MLRow) : MLValue = {
    MLValue(model.predict(x.toDoubleArray))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain() : String = {
    "Weights: " + model.weights.mkString(" ")
  }

  lazy val features: Seq[(String, Double)] = trainingTbl.schema
      .columns.drop(1)
      .zipWithIndex
      .map(c => c._1.name.getOrElse(c._2.toString))
      .zip(model.weights)
}

case class SVMParameters(
    targetCol: Int = 0,
    learningRate: Double = 0.2,
    regParam: Double = 0.0,
    maxIterations: Int = 100,
    minLossDelta: Double = 1e-5,
    minibatchFraction: Double = 1.0,
    optimizer: String = "SGD")
  extends AlgorithmParameters


object SVMAlgorithm extends Algorithm[SVMParameters] {

  def defaultParameters() = SVMParameters()

  def train(data: MLTable, params: SVMParameters): SVMModel = {

    // Initialization
    assert(data.numRows > 0)
    assert(data.numCols > 1)

    val startTime = System.currentTimeMillis

    //Run gradient descent on the data.
    val weights = SVMWithSGD.train(
      data.toRDD(params.targetCol).cache(),
      params.maxIterations,
      params.learningRate,
      params.regParam,
      params.minibatchFraction)

    val trainTime = System.currentTimeMillis - startTime

    new SVMModel(data, params, trainTime, weights)
  }

}






