package mli.interface

import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._
import org.jblas.DoubleMatrix


abstract class MLMatrix {
  //Need to figure out a better way to do this.
  var mat: DoubleMatrix
  //protected val mat: MLMatrix

  //Getting data out
  def toMLRows: Iterator[MLRow]
  def toMLVectors: Iterator[MLVector]

  //Shape
  def numCols: Index
  def numRows: Index

  //Indexing
  def apply(r: Index, c: Index): Scalar
  def apply(rows: Slice, columns: Slice): MLMatrix

  //Updating
  def update(r: Index, c: Index, v: Scalar)
  def update(rows: Slice, cols: Slice, v: MLMatrix)

  //Elementwise Algebra
  def +(r: MLMatrix): MLMatrix
  def -(r: MLMatrix): MLMatrix
  def *(r: MLMatrix): MLMatrix
  def /(r: MLMatrix): MLMatrix

  def +(r: Scalar): MLMatrix
  def -(r: Scalar): MLMatrix
  def *(r: Scalar): MLMatrix
  def /(r: Scalar): MLMatrix

  //Matrix Algebra
  def times(y: MLMatrix): MLMatrix
  def solve(y: MLMatrix): MLMatrix
  def transpose: MLMatrix

  //def norm: Double
  //def eigen: MLMatrix
  //def svd: MLMatrix
  //def rank: Int

  //Composition
  def on(y: DenseMLMatrix): MLMatrix
  def then(y: MLMatrix): MLMatrix

  //For convenience
  def dims: (Index, Index) = (numRows, numCols)
}
object MLMatrix {
  def zeros(r: Index, c: Index): MLMatrix = DenseMLMatrix.zeros(r,c)

}