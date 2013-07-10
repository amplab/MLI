package mli.ml

import scala.math
import mli.interface._

class LogisticRegressionModel( trainingTbl: MLTableLike[MLVector],
                               trainingParams: LogisticRegressionParameters,
                               trainingTime: Long,
                               val weights: MLVector)
  extends NumericModel[LogisticRegressionParameters](trainingTbl, trainingTime, trainingParams) {


  /* Predicts the label of a given data point. */
  def predict(x: MLVector) : MLValue = {
    MLDouble(LogisticRegressionAlgorithm.sigmoid(weights dot x))
  }

  def predict(tbl: MLTableLike[MLVector]): MLTableLike[MLVector] = {
    tbl.map((x: MLVector) => MLVector(Seq(predict(x).toNumber)))
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
   optimizer: String = "SGD"
) extends AlgorithmParameters


object LogisticRegressionAlgorithm extends NumericAlgorithm[LogisticRegressionParameters] with Serializable {

  def defaultParameters() = LogisticRegressionParameters()

  def sigmoid(z: Double): Double = 1.0/(1.0 + math.exp(-1.0*z))

  def train(data: MLTableLike[MLVector], params: LogisticRegressionParameters): LogisticRegressionModel = {

    // Initialization
    assert(data.numRows > 0)
    assert(data.numCols > 1)

    val d = data.numCols-1

    def gradient(row: MLVector, w: MLVector) = {

      val x = MLVector(row.slice(1,data.numCols))
      val y = row(0)
      val g = x times (sigmoid(x dot w) - y)
      g
    }

    val startTime = System.currentTimeMillis

    //Run gradient descent on the data.
    val weights = params.optimizer match {
      case "Gradient" => {
        val optparams = opt.GradientDescentParameters(MLVector.zeros(d), gradient, params.minLossDelta)
        opt.GradientDescent(data, optparams)
      }
      case "SGD" => {
        val optparams = opt.StochasticGradientDescentParameters(
          wInit = MLVector.zeros(d),
          grad=gradient,
          learningRate = params.learningRate)
        opt.StochasticGradientDescent(data, optparams)
      }
    }

    val trainTime =  System.currentTimeMillis - startTime

    new LogisticRegressionModel(data, params, trainTime, weights)
  }

}




