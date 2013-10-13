package ml.tree.node

class NodeStats(
  val gini: Option[Double] = None,
  val entropy: Option[Double] = None,
  val mean: Option[Double] = None,
  val variance: Option[Double] = None,
  val count: Option[Long] = None) extends Serializable{
  override def toString = "variance = " + variance + "count = " + count + "mean = " + mean
}
