package mli.ml

import mli.interface._

/* A generic Algorithm. Training examples are of type U, model parameters of type P. */
abstract class Algorithm[U,P] {

  def defaultParameters(): P

  def train(data: MLTableLike[U]) : Model[U,P] = train(data, defaultParameters())

  def train(data: MLTableLike[U], params: P) : Model[U,P]
}

/**
 * An abstract class for algorithms that operate on purely numeric data.
 * Expects training data as a collection of feature vectors.
 */
abstract class NumericAlgorithm[P] extends Algorithm[MLVector,P]

/**
 * Placeholder for a generic set of algorithm parameters.
 */
abstract class AlgorithmParameters