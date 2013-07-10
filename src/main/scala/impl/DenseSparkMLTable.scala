package mli.interface.impl

import mli.interface._
import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._
import spark.SparkContext
import SparkContext._


class DenseSparkMLTable ( @transient protected var rdd: spark.RDD[MLVector] ) extends MLNumericTable with Serializable {

  lazy val numCols = rdd.first.size

  lazy val numRows = rdd.count

  val schema : Schema = new Schema((0 to numCols).map(_ => new ColumnSpec(None, ColumnType.Double)))

  /**
   * Return a new RDD containing only the elements that satisfy a predicate.
   */
  def filter(f: MLVector => Boolean): DenseSparkMLTable = new DenseSparkMLTable(rdd.filter(f))

  /**
   * Return the union of this RDD and another one. Any identical elements will appear multiple
   * times (use `.distinct()` to eliminate them). For now this only works to union a DenseSparkMLTable to another.
   */
  def union(other: MLTableLike[MLVector]): DenseSparkMLTable = this
  def union(other: DenseSparkMLTable): DenseSparkMLTable = new DenseSparkMLTable(rdd.union(other.rdd))

  /**
   * Return the join of this table and another one. Any identical elements will appear multiple
   * For now this only works to union a DenseSparkMLTable to another.
   */
  def join(other: MLTableLike[MLVector], cols: Seq[Index]): DenseSparkMLTable = this
//  def join(other: DenseSparkMLTable, cols: Seq[Index]) = {
//    this
//    //Todo - extract cols from this rdd and other.rdd and call newRdd.join(newOtherRdd)
//  }


  /**
   * Return a new RDD by applying a function to all elements of this RDD.
   */
  def map(f: MLVector => MLVector): DenseSparkMLTable = new DenseSparkMLTable(rdd.map(f))


  /**
   *  Return a new RDD by first applying a function to all elements of this
   *  RDD, and then flattening the results.
   */
  def flatMap(f: MLVector => TraversableOnce[MLVector]): DenseSparkMLTable = new DenseSparkMLTable(rdd.flatMap(f))

  /**
   *  Return a value by applying a reduce function to every element of the table.
   */
  def reduce(f: (MLVector,MLVector) => MLVector): MLVector = rdd.reduce(f)



  /**
   * Return a value by applying a function to all elements of this the table and then reducing them.
   */
  def mapReduce(f: MLVector => MLVector, sum: (MLVector,MLVector) => MLVector) : MLVector =
    rdd.map(f).reduce(sum)


  /**
   * Return a new MLTable by applying a function to each partition of the table, treating that partition as a Matrix.
   * This enables block-wise computation with a more natural interface than vanilla mapPartitions.
   */
  def matrixBatchMap(f: MLMatrix => MLMatrix): DenseSparkMLTable = {
    def matrixMap(i: Iterator[MLVector]): Iterator[MLVector] = {
      val mat = DenseMLMatrix.fromVecs(i.toSeq)
      val res = f(mat)
      res.toMLVectors
    }

    val newRdd = rdd.mapPartitions(matrixMap)

    new DenseSparkMLTable(newRdd)
  }

  /**
   *  Return a new RDD by first applying a function to all elements of this
   *  RDD, and then flattening the results.
   */
  def flatMapReduce[U: ClassManifest](f: MLVector => TraversableOnce[U],
                                      sum: (U,U) => U) : U = rdd.flatMap(f).reduce(sum)


  /**
   * Reduce the ml table to a table over a specific set of columns.  Jagged
   * entries are converted to MLEmpty
   */
  def project(cols: Seq[Index]) = map(row => MLVector(cols.map(i => row(i)).toArray))

  def drop(cols: Seq[Index]) = {
    val converse = (0 until numCols).diff(cols).toArray
    project(converse)
  }

  def take(count: Int) = rdd.take(count)


  /**
   * Do not use unless for last resort.
   */

  def collect() = rdd.collect


  def print() = rdd.collect.foreach(row => println(row.mkString("\t")))

  def print(count: Int) = take(count).foreach(row => println(row.mkString("\t")))

}

object DenseSparkMLTable {
  def doubleArrayToMLRow(a: Array[Double]): MLRow = {
    val mldArray = a.map(MLDouble(_))
    DenseMLRow.fromSeq(mldArray)
  }

  def apply(rdd: spark.RDD[Array[Double]]): DenseSparkMLTable = {
    val mldRdd = rdd.map(MLVector(_))
    new DenseSparkMLTable(mldRdd)
  }
}