package ml.tree.split

/*
 * Class for storing the split predicate.
 */
class SplitPredicate(val split: Split, lessThanEqualTo: Boolean = true) extends Serializable {
  override def toString = "split = " + split.toString + ", lessThan = " + lessThanEqualTo
}
