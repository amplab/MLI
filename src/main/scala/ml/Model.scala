package mli.ml

import spark.RDD

import mli.interface._

/**
 * A generic model class. All models should inherit from this.
 * Training examples are of type T, labels of type U.
 */

abstract class Model[U,P](val trainingData: MLTableLike[U],
                        val trainingTime: Long,
                        val trainingParams: P) {

  /* Predicts the label of a given data point. */
  def predict(x: U) : MLValue
  def predict(tbl: MLTableLike[U]) : MLTableLike[U]

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain() : String

}

/**
 * An abstract class for purely numeric models. Expects Numeric input data represented as a collection of feature vectors.
 * @param trainingData
 * @param trainingTime
 * @param trainingParams
 */

abstract class NumericModel[P](trainingData: MLTableLike[MLVector], trainingTime: Long, trainingParams: P)
  extends Model[MLVector,P](trainingData, trainingTime, trainingParams)