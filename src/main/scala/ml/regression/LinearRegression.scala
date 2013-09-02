package mli.ml.classification

import mli.interface._
import mli.ml._
import org.apache.spark.mllib.regression.{LassoWithSGD,RegressionModel}

class LinearRegressionModel(
    trainingTbl: MLTable,
    trainingParams: LinearRegressionParameters,
    trainingTime: Long,
    val model: RegressionModel)
  extends Model[LinearRegressionParameters](trainingTbl, trainingTime, trainingParams) {


  /* Predicts the label of a given data point. */
  def predict(x: MLRow) : MLValue = {
    MLValue(model.predict(x.toDoubleArray))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain() : String = {
    "Weights: " + model
  }
}

case class LinearRegressionParameters(
    targetCol: Int = 0,
    learningRate: Double = 0.2,
    regParam: Double = 0.0,
    maxIterations: Int = 100,
    minLossDelta: Double = 1e-5,
    minibatchFraction: Double = 1.0,
    regStyle: String = "Lasso")
  extends AlgorithmParameters


object LinearRegressionAlgorithm extends Algorithm[LinearRegressionParameters] with Serializable {

  def defaultParameters() = LinearRegressionParameters()

  def train(data: MLTable, params: LinearRegressionParameters): LinearRegressionModel = {

    // Initialization
    assert(data.numRows > 0)
    assert(data.numCols > 1)

    val startTime = System.currentTimeMillis

    //Run gradient descent on the data.
    val sparkmodel = params.regStyle match {
      case "Lasso" => LassoWithSGD.train(
        data.toRDD(params.targetCol).cache(),
        params.maxIterations,
        params.learningRate,
        params.regParam,
        params.minibatchFraction)

//      case "Ridge" => RidgeRegression.train(
//        data.toRDD(params.targetCol),
//        params.maxIterations,
//        params.learningRate,
//        params.regParam,
//        params.minibatchFraction)
    }

    val trainTime = System.currentTimeMillis - startTime

    new LinearRegressionModel(data, params, trainTime, sparkmodel)
  }

}






