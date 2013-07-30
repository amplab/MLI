package mli.interface.impl

import mli.interface._
import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._
import spark.SparkContext
import SparkContext._


class SparkMLTable(@transient protected var rdd: spark.RDD[MLRow], var schema: Schema) extends MLTable with Serializable {
  lazy val numCols = rdd.first.size

  lazy val numRows = rdd.count

  /**
   * Return a new RDD containing only the elements that satisfy a predicate.
   */
  def filter(f: MLRow => Boolean): SparkMLTable = new SparkMLTable(rdd.filter(f), schema)

  /**
   * Return the union of this RDD and another one. Any identical elements will appear multiple
   * times (use `.distinct()` to eliminate them). For now this only works to union a DenseSparkMLTable to another.
   */
  def union(other: MLTable): SparkMLTable = this
  def union(other: SparkMLTable): SparkMLTable = {
    assert(schema == other.schema)
    new SparkMLTable(rdd.union(other.rdd), schema)
  }

  /**
   * Takes a set of columns and returns their complement in the current schema. Useful for projection, sort, join, etc.
   * @param cols
   * @param schema
   * @return
   */
  def nonCols(cols: Seq[Index], schema: Schema): Seq[Index] = schema.columns.indices.diff(cols)

  /**
   * Return the join of this table and another one. Any identical elements will appear multiple
   * For now this only works to union a DenseSparkMLTable to another.
   */
  def join(other: MLTable, cols: Seq[Index]): SparkMLTable = this

  def join(other: SparkMLTable, cols: Seq[Index]) = {

    //Parse out the key columns from each table.
    val t1 = rdd.map(row => (cols.map(row(_)), nonCols(cols, schema).map(row(_))))
    val t2 = other.rdd.map(row => (cols.map(row(_)), nonCols(cols, other.schema).map(row(_))))

    //Join the table, combine the rows, and create a new schema.
    //val newRdd = t1.join(t2).map(k => MLRow(k._1 ++ k._2._1 ++ k._2._2))
    val newRdd = t1.join(t2).map { case (a,(b,c)) => MLRow.chooseRepresentation(a ++ b ++ c)}
    lazy val newSchema = schema.join(other.schema, cols)

    new SparkMLTable(newRdd, newSchema)

  }


  /**
   * Return a new RDD by applying a function to all elements of this RDD. Schema is inferred from the results.
   * User is expected to provide a map function which produces elements of a consistent schema as output.
   */
  def map(f: MLRow => MLRow): SparkMLTable = {
    val newRdd = rdd.map(f)

    SparkMLTable.fromMLRowRdd(newRdd)
  }


  /**
   *  Return a new RDD by first applying a function to all elements of this
   *  RDD, and then flattening the results.
   */
  def flatMap(f: MLRow => TraversableOnce[MLRow]): SparkMLTable = new SparkMLTable(rdd.flatMap(f), schema)

  /**
   *  Return a value by applying a reduce function to every element of the table.
   */
  def reduce(f: (MLRow, MLRow) => MLRow): MLRow = rdd.reduce(f)

  /**
   *  Run a reduce on all values of the row, grouped by key.
   */
  def reduceBy(key: Seq[Index], f: (MLRow, MLRow) => MLRow): MLTable = {
    //val notKey = nonCols(key, schema)
    val newRdd = rdd.map(r => (r(key), r)).reduceByKey(f).map(_._2)

    SparkMLTable.fromMLRowRdd(newRdd)
  }

  def pairToRow(p: (MLRow, MLRow)) = {
    MLRow(p._1 ++ p._2)
  }

  /**
   * Sort a table based on a key.
   */
  def sortBy(key: Seq[Index], ascending: Boolean = true): MLTable = {
    //val notKey = nonCols(key, schema)
    val newRdd = rdd.map(r => (r(key), r)).sortByKey(ascending).map(_._2)
    SparkMLTable.fromMLRowRdd(newRdd)
  }

  /**
   * Return a value by applying a function to all elements of this the table and then reducing them.
   */
  def mapReduce(f: MLRow => MLRow, sum: (MLRow, MLRow) => MLRow): MLRow = rdd.map(f).reduce(sum)


  /**
   * Return a new MLTable by applying a function to each partition of the table, treating that partition as a Matrix.
   * This enables block-wise computation with a more natural interface than vanilla mapPartitions.
   */
  def matrixBatchMap(f: MLMatrix => MLMatrix): SparkMLTable = {
    def matrixMap(i: Iterator[MLRow]): Iterator[MLRow] = {
      val mat = DenseMLMatrix.fromVecs(i.map(_.toVector).toSeq)
      val res = f(mat)
      res.toMLRows
    }

    SparkMLTable.fromMLRowRdd(rdd.mapPartitions(matrixMap(_)))
  }


  /**
   * Reduce the ml table to a table over a specific set of columns.
   */
  def project(cols: Seq[Index]) = map(row => MLRow.chooseRepresentation(cols.map(i => row(i)).toSeq))

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

object SparkMLTable {
  def doubleArrayToMLRow(a: Array[Double]): MLRow = {
    val mldArray = a.map(MLValue(_))
    DenseMLRow.fromSeq(mldArray)
  }

  def apply(rdd: spark.RDD[Array[Double]]): SparkMLTable = {
    val mldRdd = rdd.map(row => MLRow.chooseRepresentation(row.map(MLValue(_))))
    lazy val schema = Schema(mldRdd.first)

    new SparkMLTable(mldRdd, schema)
  }

  def fromMLRowRdd(rdd: spark.RDD[MLRow]): SparkMLTable = {
    lazy val schema = Schema(rdd.first)

    new SparkMLTable(rdd, schema)
  }
}