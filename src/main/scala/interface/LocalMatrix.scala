package mli.interface

import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._
import org.jblas.DoubleMatrix


abstract class LocalMatrix {
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
  def apply(rows: Slice, columns: Slice): LocalMatrix

  //Updating
  def update(r: Index, c: Index, v: Scalar)
  def update(rows: Slice, cols: Slice, v: LocalMatrix)

  //Elementwise Algebra
  def +(r: LocalMatrix): LocalMatrix
  def -(r: LocalMatrix): LocalMatrix
  def *(r: LocalMatrix): LocalMatrix
  def /(r: LocalMatrix): LocalMatrix

  def +(r: Scalar): LocalMatrix
  def -(r: Scalar): LocalMatrix
  def *(r: Scalar): LocalMatrix
  def /(r: Scalar): LocalMatrix

  //Matrix Algebra
  def times(y: LocalMatrix): LocalMatrix
  def solve(y: LocalMatrix): LocalMatrix
  def transpose: LocalMatrix

  //def norm: Double
  //def eigen: MLMatrix
  //def svd: MLMatrix
  //def rank: Int

  //Composition
  def on(y: DenseMLMatrix): LocalMatrix
  def then(y: LocalMatrix): LocalMatrix

  //For convenience
  def dims: (Index, Index) = (numRows, numCols)
}
object LocalMatrix {
  def zeros(r: Index, c: Index): LocalMatrix = DenseMLMatrix.zeros(r,c)

}