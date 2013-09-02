package mli.ml

import mli.interface._

/**
 * A text model provides a way to take raw text, project it to feature space, and run predictions on it.
 * It requires a featurizer function
 */

class TextModel[P](val model: Model[P], val featurizer: MLString => MLRow) {
  def predict(p: MLString): MLValue = model.predict(featurizer(p))
}
