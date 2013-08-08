//package mli.ml.opt
//
//import mli.interface.{MLTable, MLRow}
//
//case class LBFGSParameters(
//    wInit: MLRow,
//    gradient: (MLRow, MLRow) => MLRow,
//    historySize: Int = 10) extends MLOptParameters
//
///**
// * Optimizer for performing limited memory Broyden-Fletcher-Goldfarb-Shanno optimization.
// * This is a quasi-Newton method optimizer that often leads to more precise results than
// * vanilla gradient decent methods. The tradeoff is that it is more computationally intensive
// * to compute than SGD or parallel gradient for a single iteration. It is frequently used
// * after "warm-starting" model parameters with a method like SGD.
// */
//
//object LBFGS extends MLOpt {
//  def apply(data: MLTable, params: LBFGSParameters): MLRow = {
//    lbfg(data, params.wInit, params.gradient, )
//  }
//
//  def lbfgs(data: MLTable, wInit: MLRow, gradient: (MLRow)): MLRow = {
//
//  }
//}