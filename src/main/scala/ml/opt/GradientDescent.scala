package mli.ml.opt

import mli.interface._

case class GradientDescentParameters(
  wInit: MLRow,
  gradient: (MLRow, MLRow) => MLRow,
  tol: Double = 1.0E-5) extends MLOptParameters

object GradientDescent extends MLOpt {
  def apply(dat: MLTable,
            params: GradientDescentParameters): MLRow = {
    gradientDescent(dat, params.wInit, params.gradient, params.tol)
  }

  def gradientDescent(dat: MLTable,
                      w0: MLRow,
                      grad: (MLRow, MLRow) => MLRow,
                      tol: Double): MLRow = {
    var wNew = w0
    var wOld = w0

    // Define the metric
    def metric(a: MLRow, b: MLRow): Double = ((a minus b) dot (a minus b))/(a.size)

    // Define the sum operation for gradients
    def arraySum(a: MLRow, b: MLRow): MLRow = a plus b

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
