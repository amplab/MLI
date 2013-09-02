package mli.interface.impl

import mli.interface._
import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint


class SparkMLTable(@transient protected var rdd: RDD[MLRow], inSchema: Option[Schema] = None) extends MLTable with Serializable {
  lazy val numCols = rdd.first.size

  lazy val numRows = rdd.count

  var tableSchema = inSchema

  /**
   * Return the current schema. Since schema is inferred, we wait to do this until it is explicitly asked for in the
   * common case.
   */
  def schema(): Schema = {
    tableSchema match {
      case Some(s) => s
      case None => Schema(rdd.first)
    }
  }

  /**
   * Return a new RDD containing only the elements that satisfy a predicate.
   */
  def filter(f: MLRow => Boolean): SparkMLTable = new SparkMLTable(rdd.filter(f), tableSchema)

  /**
   * Return the union of this RDD and another one. Any identical elements will appear multiple
   * times (use `.distinct()` to eliminate them). For now this only works to union a DenseSparkMLTable to another.
   */
  def union(other: MLTable): SparkMLTable = this
  def union(other: SparkMLTable): SparkMLTable = {
    assert(schema == other.schema)
    new SparkMLTable(rdd.union(other.rdd), Some(schema))
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

    new SparkMLTable(newRdd, Some(newSchema))

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
  def flatMap(f: MLRow => TraversableOnce[MLRow]): SparkMLTable = new SparkMLTable(rdd.flatMap(f))

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
   * Creates a new MLTable based on the cached version of the RDD.
   */
  def cache() = new SparkMLTable(rdd.cache(), tableSchema)

  /**
   * Sort a table based on a key.
   */
  def sortBy(key: Seq[Index], ascending: Boolean = true): MLTable = {
    //val notKey = nonCols(key, schema)
    val newRdd = rdd.map(r => (r(key), r)).sortByKey(ascending).map(_._2)
    new SparkMLTable(newRdd, tableSchema)
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
  def project(cols: Seq[Index]) = {
    //TODO - project should project schema as well.
    map(row => MLRow.chooseRepresentation(cols.map(i => row(i)).toSeq))
  }


  def take(count: Int) = rdd.take(count)

  /**
   * We support a toRDD operation here for interoperability with spark.
   */
  def toRDD(targetCol: Index = 0): RDD[LabeledPoint] = {
    val othercols = nonCols(Seq(targetCol), schema)
    rdd.map(r => LabeledPoint(r(targetCol).toNumber, r(othercols).toDoubleArray))
  }

  def toMLRowRdd(): RDD[MLRow] = rdd


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

  def apply(rdd: RDD[Array[Double]]): SparkMLTable = {
    val mldRdd = rdd.map(row => MLRow.chooseRepresentation(row.map(MLValue(_))))
    new SparkMLTable(mldRdd)
  }

  def fromMLRowRdd(rdd: RDD[MLRow]): SparkMLTable = {
    new SparkMLTable(rdd)
  }
}