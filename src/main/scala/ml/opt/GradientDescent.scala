package mli.ml.opt

import mli.interface._
import mli.interface.impl.MLNumericTable

case class GradientDescentParameters(
  wInit: MLVector,
  gradient: (MLVector, MLVector) => MLVector,
  tol: Double = 1.0E-5) extends MLOptParameters

object GradientDescent extends MLOpt {
  def apply(dat: MLNumericTable,
            params: GradientDescentParameters): MLVector = {
    gradientDescent(dat, params.wInit, params.gradient, params.tol)
  }

  def gradientDescent(dat: MLNumericTable,
                      w0: MLVector,
                      grad: (MLVector, MLVector) => MLVector,
                      tol: Double): MLVector = {
    var wNew = w0
    var wOld = w0

    // Define the metric
    def metric(a: MLVector, b: MLVector): Double = ((a minus b) dot (a minus b))/(a.size)

    // Define the sum operation for gradients
    def arraySum(a: MLVector, b: MLVector): MLVector = a plus b

    var iter = 1
    while(wNew == w0 || metric(wNew, wOld) > tol){
      val lambda = 1.0/iter
      wOld = wNew

      val totalGradient = dat.mapReduce(grad(_, wOld), arraySum(_, _))
      val avgGradient = (totalGradient times lambda) over dat.numRows
      wNew = wOld minus avgGradient
      iter += 1
    }

    wNew
  }

}
