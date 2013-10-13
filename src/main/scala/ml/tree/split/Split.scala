package ml.tree.split

/*
 * Class for storing splits -- feature index and threshold
 */
case class Split(val feature: Int, val threshold: Double) {
  override def toString = "feature = " + feature + ", threshold = " + threshold
}
