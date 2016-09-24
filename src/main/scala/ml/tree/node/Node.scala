package ml.tree.node

import org.apache.spark.rdd.RDD
import ml.tree.split.SplitPredicate
import ml.tree.Metrics._
import scala.Some

/*
 * Node trait as a template for implementing various types of nodes in the decision tree.
 */
trait Node {

  //Method for checking whether the class has any left/right child nodes.
  def isLeaf: Boolean

  //Left/Right child nodes
  def left: Node
  def right: Node

  //Depth of the node from the top node
  def depth: Int

  //RDD data as an input to the node
  def data: RDD[(Double, Array[Double])]

  //List of split predicates applied to the base RDD thus far
  def splitPredicates: List[SplitPredicate]

  // Split to arrive at the node
  def splitPredicate: Option[SplitPredicate]

  //Extract model
  def extractModel: Option[NodeModel] = {
    //Add probability logic
    if (!splitPredicate.isEmpty) {
        Some(new NodeModel(splitPredicate, left.extractModel, right.extractModel, depth, isLeaf, Some(prediction)))
    }
    else {
      Some(new NodeModel(None, None, None, depth, isLeaf, Some(prediction)))
    }
  }
  //Prediction at the node
  def prediction: Prediction
}
