package mli.impl

import mli.interface._
import mli.interface.MLTypes._
import org.jblas.{DoubleMatrix,Solve}

class DenseMLMatrix(var mat: DoubleMatrix) extends MLMatrix {

  def toMLRows: Iterator[MLRow] = {
    //TODO There must be better ways to do this.
    (0 until numRows).map(r => DenseMLRow.fromSeq(mat.getRow(r).data.map(d => MLDouble(d)).toSeq)).toIterator
  }

  def toMLVectors: Iterator[MLVector] = {
    (0 until numRows).map(r => MLVector(mat.getRow(r).data)).toIterator
  }

  def numRows: Index = mat.rows
  def numCols: Index = mat.columns

  def apply(r: Index, c: Index): Scalar = mat.get(r, c)
  def apply(rows: Slice, cols: Slice): MLMatrix = new DenseMLMatrix(mat.get(cols.toArray, rows.toArray))

  def update(r: Index, c: Index, v: Scalar) = mat.put(r, c, v)
  def update(rows: Slice, cols: Slice, v: MLMatrix) = {
    //Jblas Matrices are row-major, so this shoudl be faster in the congiguous case.
    for (c <- cols) {
      for (r <- rows) {
        mat.put(r, c, v(r, c))
      }
    }
  }

  def *(y: MLMatrix) = new DenseMLMatrix(mat.mul(y.mat))
  def +(y: MLMatrix) = new DenseMLMatrix(mat.add(y.mat))
  def -(y: MLMatrix) = new DenseMLMatrix(mat.sub(y.mat))
  def /(y: MLMatrix) = new DenseMLMatrix(mat.div(y.mat))

  def *(y: Scalar) = new DenseMLMatrix(mat.mul(y))
  def +(y: Scalar) = new DenseMLMatrix(mat.add(y))
  def -(y: Scalar) = new DenseMLMatrix(mat.sub(y))
  def /(y: Scalar) = new DenseMLMatrix(mat.sub(y))

  def solve(y: MLMatrix) = new DenseMLMatrix(Solve.solve(mat, y.mat))
  def times(y: MLMatrix) = new DenseMLMatrix(mat.mmul(y.mat))
  def transpose = new DenseMLMatrix(mat.transpose())

  //TODO need to decide on types for these.
  //def norm = breeze.linalg.norm(mat)
  //def eigen = new DenseMLMatrix(breeze.linalg.eig(mat))
  //def svd = new DenseMLMatrix(breeze.linalg.svd(mat))

  //Composition
  def on(y: DenseMLMatrix): MLMatrix = new DenseMLMatrix(DoubleMatrix.concatVertically(mat, y.mat))
  def then(y: MLMatrix): MLMatrix = new DenseMLMatrix(DoubleMatrix.concatHorizontally(mat, y.mat))
}

/**
 * Contains facilities for creating a Dense Matrix from rows (either a Seq of MLRow or MLVectors).
 */
object DenseMLMatrix {
  def apply(rows: DenseMLRow*) = fromSeq(rows.toSeq)
  def fromSeq(rows: Seq[DenseMLRow]) = {
    val dat = rows.map(_.toDoubleArray).toArray
    new DenseMLMatrix(new DoubleMatrix(dat))
  }

  def fromVecs(rows: Seq[MLVector]) = {
    val dat = rows.map(_.toArray).toArray
    new DenseMLMatrix(new DoubleMatrix(dat))
  }

  def zeros(rows: Index, cols: Index) = new DenseMLMatrix(DoubleMatrix.zeros(rows, cols))
  def eye(n: Index) = new DenseMLMatrix(DoubleMatrix.eye(n))

}
