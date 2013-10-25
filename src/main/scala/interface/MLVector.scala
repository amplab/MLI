package mli.interface

import org.jblas.DoubleMatrix
import mli.impl.DenseMLMatrix


trait MLVectorable {
  def toVector: MLVector
}

/**
 * A numerical vector supporting various mathematical operations efficiently.
 */
class MLVector(val data: DoubleMatrix) extends IndexedSeq[Double] with Serializable {
  override def toString: String = data.toString

  //We need this for slice syntax.
  def apply(idx: Int): Double = data.get(idx)

  //Here we provide a few useful vector ops.
  def dot(other: MLVector): Double = data dot other.data
  def times(other: MLVector): MLVector = new MLVector(data.mul(other.data))
  def plus(other: MLVector): MLVector = new MLVector(data.add(other.data))
  def minus(other: MLVector): MLVector = new MLVector(data.sub(other.data))
  def over(other: MLVector): MLVector = new MLVector(data.div(other.data))

  def times(other: Double): MLVector = new MLVector(data.mul(other))
  def plus(other: Double): MLVector = new MLVector(data.add(other))
  def minus(other: Double): MLVector = new MLVector(data.sub(other))
  def over(other: Double): MLVector = new MLVector(data.div(other))


  //def outer(other: MLVector): MLMatrix = new MLMatrix(transpose(data) * data)
  def length = data.length
  def sum: Double = data.sum

  def toMatrix: LocalMatrix = new DenseMLMatrix(new DoubleMatrix(data.data).transpose)
  def toArray = data.data
  //val row = MLRow(this)
}


object MLVector {
  def apply(data: Iterator[Double]): MLVector = apply(data.toArray)
  def apply(data: IndexedSeq[Double]): MLVector = apply(data.toArray)
  def apply(data: Seq[Double]): MLVector = apply(data.toArray)
  def apply(data: Array[Double]): MLVector = new MLVector(new DoubleMatrix(data))
  def apply(data: Array[MLValue]): MLVector = apply(data.map(_.toNumber))

  //Returns a zero vector of length D.
  def zeros(d: Int): MLVector = apply(new Array[Double](d))
  implicit def vectorToRow(from: MLVector): MLRow = MLRow(from)//from.row
}