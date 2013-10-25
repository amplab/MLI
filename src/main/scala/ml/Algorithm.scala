package mli.ml

import mli.interface._
import mli.interface.impl.MLNumericTable

/* A generic Algorithm. Training examples are of type U, model parameters of type P. */
abstract class Algorithm[P] extends Serializable{

  def defaultParameters(): P

  def train(data: MLTable) : Model[P] = train(data, defaultParameters())

  def train(data: MLTable, params: P) : Model[P]
}

abstract class NumericAlgorithm[P] extends Serializable{

  def defaultParameters(): P

  def train(data: MLNumericTable) : NumericModel[P] = train(data, defaultParameters())

  def train(data: MLNumericTable, params: P) : NumericModel[P]
}

/**
 * Placeholder for a generic set of algorithm parameters.
 */
abstract class AlgorithmParameters extends Serializable