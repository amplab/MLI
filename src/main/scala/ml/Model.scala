package mli.ml

import mli.interface._
import mli.interface.impl.MLNumericTable

/**
 * A generic model class. All models should inherit from this.
 * Training examples are of type T, labels of type U.
 */

abstract class Model[P](val trainingData: MLTable,
                        val trainingTime: Long,
                        val trainingParams: P) extends Serializable{

  /* Predicts the label of a given data point. */
  def predict(x: MLRow) : MLValue
  def predict(tbl: MLTable) : MLTable = {
    tbl.map((x: MLRow) => MLRow.chooseRepresentation(Seq(predict(x))))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain(): String

}

abstract class NumericModel[P](val trainingData: MLNumericTable,
                               val trainingTime: Long,
                               val trainingParams: P) extends Serializable {
  /* Predicts the label of a given data point. */
  def predict(x: MLVector) : MLValue
  def predict(tbl: MLNumericTable) : MLNumericTable = {
    tbl.map((x: MLVector) => MLVector(Seq(predict(x).toNumber)))
  }

  /**
   * Provides a user-friendly explanation of this model.
   * For example, plots or console output.
   */
  def explain(): String
}
