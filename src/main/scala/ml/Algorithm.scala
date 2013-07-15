package mli.ml

import mli.interface._

/* A generic Algorithm. Training examples are of type U, model parameters of type P. */
abstract class Algorithm[P] {

  def defaultParameters(): P

  def train(data: MLTable) : Model[P] = train(data, defaultParameters())

  def train(data: MLTable, params: P) : Model[P]
}

/**
 * Placeholder for a generic set of algorithm parameters.
 */
abstract class AlgorithmParameters