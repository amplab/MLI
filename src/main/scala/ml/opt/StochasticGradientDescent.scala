package mli.ml.opt

import mli.interface._
import mli.interface.impl.MLNumericTable

/**
 * Parameters for the Stochastic Gradient Descent optimizer.
 *
 * @param learningRate Scaling constant at each gradient update.
 * @param wInit Initial model weights.
 * @param maxIter Maximum number of iterations to perform before returning a model.
 * @param eps Parameter used to check for model convergence after each pass over the data.
 * @param grad Gradient function - expecting a data vector and weight vector and returning the gradient.
 */
case class StochasticGradientDescentParameters(
                                                learningRate: Double = 1e-2,
                                                wInit: MLVector,
                                                maxIter: Int = 100,
                                                eps: Double = 1e-6,
                                                grad: (MLVector, MLVector) => MLVector
                                                ) extends MLOptParameters


object StochasticGradientDescent extends MLOpt with Serializable {
  /**
   * Main entry point for Stochastic Gradient Descent (SGD) Optimizer.
   * This optimizer performs SGD at each partition locally, and averages parameters from
   * each partition together after each pass over the data.
   *
   * @param data A numeric table of features and a label.
   * @param params Parameters for the gradient descent algorithm.
   * @return
   */
  def apply(data: MLNumericTable, params: StochasticGradientDescentParameters): MLVector = {
    runSGD(data, params.wInit, params.learningRate, params.grad, params.maxIter, params.eps)
  }

  def runSGD(data: MLNumericTable,
             wInit: MLVector,
             learningRate: Double,
             grad: (MLVector, MLVector) => MLVector,
             maxIter: Int,
             eps: Double): MLVector = {
    var weights = wInit
    var weightsOld = weights

    val n = data.numRows
    var i = 0
    var diff = 1e6

    //Main loop of SGD. Calls local SGD and averages parameters. Checks for convergence after each pass.
    while(i < maxIter && diff > eps) {
      //weights = data.map(sgdStep(_, weights, learningRate, grad)).reduce(_ plus _) over n
      weights = data.matrixBatchMap(localSGD(_, weights, learningRate, grad)).reduce(_ plus _) over n

      diff = ((weights minus weightsOld) dot (weights minus weightsOld))
      println(diff)
      weightsOld = weights
      i+=1
    }

    weights

  }

  def sgdStep(data: MLVector, weights: MLVector, lambda: Double, gradientFunction: (MLVector, MLVector) => MLVector): MLVector = {
    gradientFunction(data, weights) times lambda
  }

  /**
   * Locally runs SGD on each partition of data. Sends results back to master after each pass.
   */
  def localSGD(data: LocalMatrix, weights: MLVector, lambda: Double, gradientFunction: (MLVector, MLVector) => MLVector): LocalMatrix = {
    //Set local weights.
    var loc = weights

    //For each row in the matrix.
    for (i <- data.toMLVectors) {
      //Compute the gradient.
      val grad = gradientFunction(i, loc)

      //Update according to the learning rate.
      loc = loc minus (grad times lambda)
    }

    //Return the results.
    loc.toMatrix
  }
}
