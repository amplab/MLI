package mli.ml.opt

import mli.interface._

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
    wInit: MLRow,
    maxIter: Int = 100,
    eps: Double = 1e-6,
    grad: (MLRow, MLRow) => MLRow
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
  def apply(data: MLTable, params: StochasticGradientDescentParameters): MLRow = {
    runSGD(data, params.wInit, params.learningRate, params.grad, params.maxIter, params.eps)
  }

  def runSGD(
              data: MLTable,
              wInit: MLVector,
              learningRate: Double,
              grad: (MLRow, MLRow) => MLRow,
              maxIter: Int,
              eps: Double
            ): MLRow = {

    //Initialize the model weights and set data size.
    var weights = wInit
    var weightsOld = weights

    val n = data.numRows
    var i = 0
    var diff = 1e6

    //Main loop of SGD. Calls local SGD and averages parameters. Checks for convergence after each pass.
    while(i < maxIter && diff > eps) {
      weights = data.matrixBatchMap(localSGD(_, weights, learningRate, grad)).reduce(_ plus _) over n

      diff = ((weights minus weightsOld) dot (weights minus weightsOld))
      i+=1
    }

    weights
  }

  /**
   * Locally runs SGD on each partition of data. Sends results back to master after each pass.
   */
  def localSGD(data: MLMatrix, weights: MLVector, lambda: Double, gradientFunction: (MLRow, MLRow) => MLRow): MLMatrix = {
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
