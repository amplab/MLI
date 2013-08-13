package mli.ml.classification

import mli.ml._
import scala.math
import mli.interface._

class LogisticRegressionModel(
    trainingTbl: MLTable,
    trainingParams: LogisticRegressionParameters,
    trainingTime: Long,
    val weights: MLRow)
  extends Model[LogisticRegressionParameters](trainingTbl, trainingTime, trainingParams) with Serializable {


  /* Predicts the label of a given data point. */
  def predict(x: MLRow) : MLValue = {
    MLValue(LogisticRegressionAlgorithm.sigmoid(weights dot x))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain() : String = {
    "Weights: " + weights.toString
  }
}

case class LogisticRegressionParameters(
    learningRate: Double = 0.2,
    maxIterations: Int = 100,
    minLossDelta: Double = 1e-5,
    optimizer: String = "SGD")
  extends AlgorithmParameters


object LogisticRegressionAlgorithm extends Algorithm[LogisticRegressionParameters] with Serializable {

  def defaultParameters() = LogisticRegressionParameters()

  def sigmoid(z: Double): Double = 1.0/(1.0 + math.exp(-1.0*z))

  def train(data: MLTable, params: LogisticRegressionParameters): LogisticRegressionModel = {

    // Initialization
    assert(data.numRows > 0)
    assert(data.numCols > 1)

    val d = data.numCols-1

    def gradient(row: MLRow, w: MLRow): MLRow = {

      val x = MLVector(row.slice(1,row.length))
      val y = row(0).toNumber
      val g = x times (sigmoid(x dot w) - y)
      g
    }

    val startTime = System.currentTimeMillis

    //Run gradient descent on the data.
    val weights = params.optimizer match {
      case "Gradient" => {
        val optParams = opt.GradientDescentParameters(MLVector.zeros(d), gradient, params.minLossDelta)
        opt.GradientDescent(data, optParams)
      }
      case "SGD" => {
        val optParams = opt.StochasticGradientDescentParameters(
          wInit = MLVector.zeros(d),
          grad = gradient,
          learningRate = params.learningRate)
        opt.StochasticGradientDescent(data, optParams)
      }
    }

    val trainTime =  System.currentTimeMillis - startTime

    new LogisticRegressionModel(data, params, trainTime, weights)
  }

}




