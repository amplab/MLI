package mli.ml.opt

import mli.interface.{MLMatrix, MLTableLike, MLVector}

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
  def apply(data: MLTableLike[MLVector], params: StochasticGradientDescentParameters): MLVector = {
    runSGD(data, params.wInit, params.learningRate, params.grad, params.maxIter, params.eps)
  }

  def runSGD(
              data: MLTableLike[MLVector],
              wInit: MLVector,
              learningRate: Double,
              grad: (MLVector, MLVector) => MLVector,
              maxIter: Int,
              eps: Double
            ): MLVector = {

    //Initialize the model weights and set data size.
    var weights = wInit
    var weightsOld = weights

    val n = data.numRows
    var i = 0
    var diff = 1e6

    //Main loop of SGD. Calls local SGD and averages parameters. Checks for convergence after each pass.
    while(i < maxIter && diff > eps) {
      weightsOld = weights
      weights = data.matrixBatchMap(localSGD(_, weights, learningRate, grad)).reduce(_ plus _) over n

      diff = ((weights minus weightsOld) times (weights minus weightsOld)).sum
      i+=1
    }

    weights
  }

  /**
   * Locally runs SGD on each partition of data. Sends results back to master after each pass.
   */
  def localSGD(data: MLMatrix, weights: MLVector, lambda: Double, gradientFunction: (MLVector, MLVector) => MLVector): MLMatrix = {
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
