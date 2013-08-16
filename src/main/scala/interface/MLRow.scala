package mli.interface

import mli.interface.MLTypes._
import scala.collection.immutable.TreeMap
import scala.collection.mutable.ArrayBuffer
import scala.collection._
import generic.{HasNewBuilder, SeqFactory, CanBuildFrom}
import scala._
import scala.collection.immutable
import scala.collection.IndexedSeq
import scala.Some
import scala.Iterator
import scala.collection.Seq
import scala.collection.TraversableOnce
import scala.Some

/**
 * A single row composed of zero or more columns.  MLRow is typically used in
 * an MLTable or as a standalone parameter vector.
 *
 * MLVector operations supported by this type are generally fast.  Scala
 * collections operations inherited from Seq, like map(), are provided for
 * convenience but should generally not be used in performance-critical code.
 * When an MLRow could be large and sparse, prefer MLVector operations or
 * operations that act on non-zero entries, such as the iterator nonZeros().
 */
trait MLRow extends IndexedSeq[MLValue]
    with IndexedSeqLike[MLValue, MLRow]
    with MLVectorable
    with Ordered[MLRow]
    with Serializable {

  /** Supports indexing by a sequence of indices. */
  def apply(inds: Seq[Int]): MLRow = MLRow(inds.map(this(_)):_*)

  def drop(cols: Seq[Int]) = {
    val converse = (0 until this.length).diff(cols).toArray
    apply(converse)
  }

  /** An iterator through the nonzero rows ((index, value) pairs) of this row. */
  def nonZeros(): Iterator[(Int, MLValue)]

  /** A method for building the implementation from a sequence of MLValues. */
  def newMlRowBuilder: mutable.Builder[MLValue, MLRow]

  implicit def vectorToRow(v: MLVector): MLRow = MLRow.chooseRepresentation(v)

  override protected[this] def newBuilder = MLRow.newBuilder

  override def toString = this.mkString("\t")

  /** Implements dense lexicographic ordering on an MLRow */
  def compare(that: MLRow): Int = {
    for ((thisOne,thatOne) <- this.zip(that)) {
      if(thisOne != thatOne) {
        if (thisOne > thatOne) return 1
        else return -1
      }
    }
    return 0
  }

  def toDoubleArray: Array[Double]
}

object MLRow {
  private val MIN_SIZE_FOR_SPARSE_REPRESENTATION = 1000
  private val MAX_DENSITY_FOR_SPARSE_REPRESENTATION = .5

  def apply(row: MLValue*) = chooseRepresentation(row.toSeq)
  def apply(row: MLVector) = chooseRepresentation(row.data.data.map(MLValue(_)).toSeq)

  /**
   * Choose a reasonable sparse or dense representation for @row.
   *
   * @param row should contain only numeric values.
   */
  def chooseRepresentation(row: Seq[MLValue]): MLRow = {
    val len = row.length
    if (len >= MIN_SIZE_FOR_SPARSE_REPRESENTATION
        && row.count(_.toNumber != 0) < len * MAX_DENSITY_FOR_SPARSE_REPRESENTATION ) {
      SparseMLRow.fromNumericSeq(row)
    } else {
      DenseMLRow.fromSeq(row.toIndexedSeq)
    }
  }

  /** @see chooseRepresentation(row: Seq) */
  def chooseRepresentation(row: Array[MLValue]): MLRow = chooseRepresentation(row.toSeq)

  def newBuilder: mutable.Builder[MLValue, MLRow] = new ArrayBuffer[MLValue]().mapResult({array => chooseRepresentation(array.toSeq)})

  implicit def canBuildFrom: CanBuildFrom[MLRow, MLValue, MLRow] = new CanBuildFrom[MLRow, MLValue, MLRow] {
    override def apply() = newBuilder
    // Allow subclasses to define their own builders.
    override def apply(from: MLRow) = from.newMlRowBuilder
  }

  implicit def rowToVector(from: MLRow): MLVector = from.toVector
}


/**
 * An implementation of MLRow targeted for rows with mostly interesting (e.g.
 * non-zero) values.
 *
 * @param row the actual values of the row.
 */
class DenseMLRow(private val row: immutable.IndexedSeq[MLValue]) extends MLRow {
  override def length = row.length

  override def apply(index: Int) = row.apply(index)

  override def iterator = row.iterator

  override def nonZeros = (0 until length).map({index => (index, row(index))}).filter(_._2.toNumber != 0).iterator

  override def newMlRowBuilder = new ArrayBuffer[MLValue]().mapResult({array => DenseMLRow.fromSeq(array.toSeq)})

  lazy val vec = MLVector(row.toArray)

  override implicit def toVector = vec

  def toDoubleArray = row.map(_.toNumber).toArray
}

object DenseMLRow {
  def apply(row: MLValue*) = fromSeq(row.toSeq)

  def fromSeq(row: Seq[MLValue]) = new DenseMLRow(row.toIndexedSeq)
}

/**
 * A simple implementation of a sparse row.  Could use some actual
 * optimization for vector ops.
 *
 * @param sparseElements is a sparse representation of this row, a map from column
 *                 indices to MLValues.  Indices not present in this map will
 *                 be considered empty.
 * @param trueLength is the actual length of the row, including empty columns.
 * @param emptyValue is the default value for elements not in @param elements.
 */
class SparseMLRow private(
    private val sparseElements: immutable.SortedMap[Int, MLValue],
    private val trueLength: Int,
    private val emptyValue: MLValue) extends MLRow {
  override def length = trueLength

  override def apply(index: Int): MLValue = sparseElements.get(index) match {
    case None => emptyValue
    case Some(value) => value
  }

  override def iterator = (0 until trueLength).map(index => apply(index)).iterator

  override def nonZeros = sparseElements.iterator

  lazy val vec = MLVector(iterator.toArray)
  override implicit def toVector = MLVector(iterator.toArray)
  def toDoubleArray = this.toVector.data.data

  //TODO: Need to do some performance testing here.
  override def newMlRowBuilder = new ArrayBuffer[MLValue]().mapResult({array =>
    val n = TreeMap[Int, MLValue](array.zipWithIndex.filter(_._1 != emptyValue).map(x => (x._2,x._1)):_*)
    new SparseMLRow(n, array.length, emptyValue)
  })
}


object SparseMLRow {
  def fromSparseCollection(elements: TraversableOnce[(Int, MLValue)], trueLength: Int, emptyValue: MLValue)
    = new SparseMLRow(TreeMap.empty[Int, MLValue] ++ elements, trueLength, emptyValue)

  def fromNumericSeq(row: Seq[MLValue]) = {
    val nonZeros: Seq[(Int, MLValue)] = (0 until row.length).zip(row).filter(_._2.toNumber != 0)
    new SparseMLRow(TreeMap.empty[Int, MLValue] ++ nonZeros, row.length, MLValue(0.0))
  }
}
