package ml.tree.node

/*
 * Class used to store the prediction values at each node of the tree.
 */
class Prediction(val prob: Double, val distribution: Map[Double, Double]) extends Serializable	{
  override def toString = { "probability = " + prob + ", distribution = " + distribution }
}
